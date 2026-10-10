#!/usr/bin/env python3
"""gaze-app: 目線だけで動く仮想カーソルのデスクトップアプリ。

ウィンドウを出して、その中だけで完結する。**実カーソルは一切操作しない**（ジャックしない）。
視線の推定は gaze_core.py（数値）と gaze_vision.py（カメラ/顔/虹彩）が担当し、
このファイルは Tkinter の画面と状態遷移だけを持つ。

    python3 gaze_app.py             # 全画面で起動
    python3 gaze_app.py --windowed  # ウィンドウ表示（動作確認用）
    python3 gaze_app.py --self-test # カメラと検出の確認だけして終了
    python3 gaze_app.py --fetch-model  # dlib の学習済みモデルを取得

キー:  Esc=終了 / c=キャリブレーション / t=追跡の入切 / d=デバッグ表示 / m=モード切替 / f=全画面切替
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import tkinter as tk
from tkinter import font as tkfont

import gaze_core as core

# 画面まわりの色（自分のサイトと同じ落ち着いた配色に揃える）
BG = "#141017"
TEXT = "#e6e6ea"
MUTED = "#8d8d97"
BORDER = "#2a2530"
ACCENT = "#d98fae"

# キャリブレーションの点（最初の中央で「正面の基準」を取る）
CALIBRATION_TARGETS = [
    (0.50, 0.50),
    (0.10, 0.10), (0.50, 0.10), (0.90, 0.10),
    (0.10, 0.50), (0.90, 0.50),
    (0.10, 0.90), (0.50, 0.90), (0.90, 0.90),
    (0.30, 0.10), (0.70, 0.10),
    (0.30, 0.90), (0.70, 0.90),
]
SAMPLES_PER_TARGET = 40
SETTLE_MS = 700
SAMPLE_TIMEOUT_MS = 3000
MIN_SAMPLES = 10

# キャリブ結果の世代。壊れた推定で保存された古いモデルを読み込まないようにする。
MODEL_VERSION = 2

TARGET_TILES = ["A", "B", "C", "D", "E"]
MODES = [("both", "視線+頭"), ("gaze", "視線のみ"), ("head", "頭のみ")]


def app_home() -> str:
    home = os.environ.get("GAZE_APP_HOME", os.path.expanduser("~/.cache/gaze-app"))
    os.makedirs(home, exist_ok=True)
    return home


class GazeApp:
    def __init__(self, root: tk.Tk, camera_index: int = 0, windowed: bool = False) -> None:
        self.root = root
        self.camera_index = camera_index
        self.windowed = windowed
        self.mode = core.MODE_BOTH

        root.title("gaze-cursor")
        root.configure(bg=BG)
        if not windowed:
            root.attributes("-fullscreen", True)
        else:
            root.geometry("1280x800")

        self.canvas = tk.Canvas(root, bg=BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.mono = tkfont.Font(family="monospace", size=11)
        self.mono_small = tkfont.Font(family="monospace", size=9)
        self.label_font = tkfont.Font(family="sans-serif", size=12)

        self.width = root.winfo_screenwidth()
        self.height = root.winfo_screenheight()

        # --- 推定・制御の状態 ---
        self.model: core.GazeModel | None = None
        self.filters: dict[str, core.OneEuroFilter] = {}
        self.medians: dict[str, core.MedianFilter] = {}
        self.eye_median = core.MedianFilter(61)
        self.last_eye = None
        self.last_iris = None
        self.rejected = 0
        self.iris_missing = 0
        self.out_of_range = 0
        self.frames = 0
        self.gaze = (0.5, 0.5)
        self.gaze_px = (0.0, 0.0)
        self.gaze_ever = False
        self.openness = 0.0
        self.fit_error = 0.0
        self.calib_error = 0.0
        self.pose = (0.0, 0.0)
        self.controller = core.CursorController({"deadzone_px": 40.0})
        self.cursor = (0.0, 0.0)
        self.debug = False
        self.status_line = ""
        self.hint = "START を押してカメラを起動してください"

        self.calib: dict | None = None
        self.photo = None
        self.camera_updated_at = 0.0
        self.camera_view = None   # (表示幅, 表示高さ, 倍率) 検出点の座標変換に使う
        self.tile_counts = {key: 0 for key in TARGET_TILES}

        self.hud_items: dict[str, int] = {}
        self.canvas_items: dict[str, int] = {}
        self.buttons: list[dict] = []
        self.tile_rects: list[dict] = []

        self.worker = None
        self.running = True
        self.mode_index = 0

        self._build_static_items()
        self.canvas.bind("<Button-1>", self._on_click)
        root.bind("<Key>", self._on_key)
        root.bind("<Configure>", lambda event: self._layout())

        self.load_model()
        self.reset_filters()
        self._layout()
        self.root.after(16, self._ui_tick)
        self.root.after(10, self._control_tick)

    # ------------------------------------------------------------------
    # 起動・停止
    # ------------------------------------------------------------------

    def start_camera(self) -> None:
        if self.worker is not None:
            return
        try:
            import gaze_vision

            self.worker = gaze_vision.VisionWorker(self.camera_index)
        except FileNotFoundError as error:
            self.hint = str(error)
            return
        except Exception as error:  # dlib の読み込み失敗など
            self.hint = f"視線検出を準備できませんでした: {error}"
            return
        self.worker.start()
        self.hint = "カメラを起動しました。c でキャリブレーション、t で追跡開始"

    def shutdown(self) -> None:
        self.running = False
        if self.worker is not None:
            self.worker.stop()
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    # ------------------------------------------------------------------
    # 画面の組み立て
    # ------------------------------------------------------------------

    def _build_static_items(self) -> None:
        canvas = self.canvas
        for key in TARGET_TILES:
            rect = canvas.create_rectangle(0, 0, 0, 0, outline=BORDER, fill="#1a1520", width=1)
            label = canvas.create_text(0, 0, text=key, fill=MUTED, font=self.mono)
            count = canvas.create_text(0, 0, text="0", fill=MUTED, font=self.mono_small)
            self.tile_rects.append({"key": key, "rect": rect, "label": label, "count": count})

        self.canvas_items["cursor_shadow"] = canvas.create_oval(0, 0, 0, 0, fill="#0b0910", outline="")
        self.canvas_items["cursor_ring"] = canvas.create_oval(0, 0, 0, 0, fill="", outline=TEXT, width=2)
        self.canvas_items["cursor"] = canvas.create_oval(0, 0, 0, 0, fill=ACCENT, outline="")
        self.canvas_items["dwell"] = canvas.create_arc(0, 0, 0, 0, start=90, extent=0,
                                                       style="arc", outline=TEXT, width=2)
        self.canvas_items["gaze"] = canvas.create_line(0, 0, 0, 0, fill=MUTED, state="hidden")
        self.canvas_items["gaze2"] = canvas.create_line(0, 0, 0, 0, fill=MUTED, state="hidden")
        self.canvas_items["calib"] = canvas.create_oval(0, 0, 0, 0, outline=MUTED, width=1, state="hidden")
        self.canvas_items["calib_dot"] = canvas.create_oval(0, 0, 0, 0, fill=ACCENT, outline="", state="hidden")
        self.canvas_items["calib_ring"] = canvas.create_arc(0, 0, 0, 0, start=90, extent=0,
                                                            style="arc", outline=ACCENT, width=3, state="hidden")
        self.canvas_items["calib_text"] = canvas.create_text(0, 0, text="", fill=MUTED,
                                                             font=self.label_font, state="hidden")
        for name in ("STATE", "GAZE", "POSE", "FIT", "RANGE", "OPEN", "LAT", "MODE"):
            self.hud_items[name] = canvas.create_text(0, 0, text="", anchor="e", fill=MUTED, font=self.mono)
        self.canvas_items["hint"] = canvas.create_text(0, 0, text="", fill=TEXT, font=self.label_font)
        self.canvas_items["status"] = canvas.create_text(0, 0, text="", anchor="w", fill=MUTED,
                                                         font=self.mono_small)
        # カメラは画面全体にうっすら重ねる（Tk に透明度が無いので画像側を暗く合成する）
        self.canvas_items["camera"] = canvas.create_image(0, 0, anchor="center", state="hidden")
        for index in range(2):
            self.canvas_items[f"mark_iris{index}"] = canvas.create_oval(
                0, 0, 0, 0, outline="#6f6ff5", width=2, state="hidden")
            self.canvas_items[f"mark_note{index}"] = canvas.create_text(
                0, 0, text="", anchor="w", fill="#c8dc50", font=self.mono_small, state="hidden")

        for name, text in (("start", "START"), ("calibrate", "CALIBRATE"), ("track", "TRACKING"),
                           ("mode", "MODE"), ("debug", "DEBUG"), ("reset", "RESET"), ("exit", "EXIT")):
            rect = canvas.create_rectangle(0, 0, 0, 0, outline=BORDER, fill="")
            label = canvas.create_text(0, 0, text=text, fill=TEXT, font=self.label_font)
            self.buttons.append({"name": name, "rect": rect, "label": label})

    def _layout(self) -> None:
        width = self.canvas.winfo_width() or self.width
        height = self.canvas.winfo_height() or self.height
        if width < 50 or height < 50:
            return
        self.width, self.height = width, height
        canvas = self.canvas

        size = max(76, int(min(width, height) * 0.13))
        gap = int(size * 0.28)
        total = len(self.tile_rects) * size + (len(self.tile_rects) - 1) * gap
        left = (width - total) // 2
        top = (height - size) // 2
        for index, tile in enumerate(self.tile_rects):
            x0 = left + index * (size + gap)
            canvas.coords(tile["rect"], x0, top, x0 + size, top + size)
            canvas.coords(tile["label"], x0 + size / 2, top + size * 0.42)
            canvas.coords(tile["count"], x0 + size / 2, top + size * 0.66)

        y = 18
        for name in ("STATE", "GAZE", "POSE", "FIT", "RANGE", "OPEN", "LAT", "MODE"):
            canvas.coords(self.hud_items[name], width - 24, y)
            y += 17

        bw, bh, bgap = 106, 34, 9
        total_b = len(self.buttons) * bw + (len(self.buttons) - 1) * bgap
        bx = (width - total_b) // 2
        by = height - 56
        for index, button in enumerate(self.buttons):
            x0 = bx + index * (bw + bgap)
            canvas.coords(button["rect"], x0, by, x0 + bw, by + bh)
            canvas.coords(button["label"], x0 + bw / 2, by + bh / 2)

        canvas.coords(self.canvas_items["hint"], width / 2, height - 84)
        canvas.coords(self.canvas_items["status"], 24, height - 76)
        canvas.coords(self.canvas_items["camera"], width / 2, height / 2)

    # ------------------------------------------------------------------
    # 入力
    # ------------------------------------------------------------------

    def _on_key(self, event) -> None:
        key = event.keysym.lower()
        if key == "escape":
            self.shutdown()
        elif key == "c":
            self.start_calibration()
        elif key == "t":
            self.toggle_tracking()
        elif key == "d":
            self.debug = not self.debug
        elif key in ("m", "period"):
            self.cycle_mode()
        elif key == "f":
            self.windowed = not self.windowed
            self.root.attributes("-fullscreen", not self.windowed)
            self._layout()

    def _on_click(self, event) -> None:
        for button in self.buttons:
            x0, y0, x1, y1 = self.canvas.coords(button["rect"])
            if x0 <= event.x <= x1 and y0 <= event.y <= y1:
                self._press(button["name"])
                return

    def _press(self, name: str) -> None:
        if name == "start":
            self.start_camera()
        elif name == "calibrate":
            self.start_calibration()
        elif name == "track":
            self.toggle_tracking()
        elif name == "mode":
            self.cycle_mode()
        elif name == "debug":
            self.debug = not self.debug
        elif name == "reset":
            self.tile_counts = {key: 0 for key in TARGET_TILES}
        elif name == "exit":
            self.shutdown()

    def cycle_mode(self) -> None:
        self.mode_index = (self.mode_index + 1) % len(MODES)
        self.mode = MODES[self.mode_index][0]
        self.hint = {
            core.MODE_BOTH: "視線と頭の向きを統合して動かします",
            core.MODE_GAZE: "視線だけで動かします（頭の向きは使いません）",
            core.MODE_HEAD: "頭の向きだけで動かします（視線は使いません）",
        }[self.mode]

    # ------------------------------------------------------------------
    # キャリブレーション
    # ------------------------------------------------------------------

    def start_calibration(self) -> None:
        if self.worker is None:
            self.hint = "先に START でカメラを起動してください"
            return
        self.stop_tracking("キャリブレーションのため停止しました")
        now_ms = time.monotonic() * 1000.0
        self.debug = True   # キャリブ中は写りを確認できるようカメラを薄く出す
        self.calib = {
            "index": 0,
            "samples": [[] for _ in CALIBRATION_TARGETS],
            "phase": "settle",
            "until": now_ms + SETTLE_MS,
        }
        self.hint = "出てくる点を順に注視してください"

    def _advance_calibration(self, observation, now_ms: float) -> None:
        calib = self.calib
        index = calib["index"]
        if calib["phase"] == "settle":
            if now_ms >= calib["until"]:
                calib["phase"] = "collect"
                calib["until"] = now_ms + SAMPLE_TIMEOUT_MS
            return

        if observation is not None:
            calib["samples"][index].append((observation["landmarks"], observation["irises"]))

        collected = len(calib["samples"][index])
        if collected < SAMPLES_PER_TARGET and not (now_ms >= calib["until"] and collected >= MIN_SAMPLES):
            return

        if collected < MIN_SAMPLES:
            self.calib = None
            self.hint = f"点 {index + 1} で顔を検出できませんでした。明るさと距離を確認してください"
            return

        calib["index"] += 1
        if calib["index"] < len(CALIBRATION_TARGETS):
            calib["phase"] = "settle"
            calib["until"] = now_ms + SETTLE_MS
            return
        self._finish_calibration()

    def _finish_calibration(self) -> None:
        samples_per_target = self.calib["samples"]
        self.calib = None

        center = samples_per_target[0]
        count = len(center)
        reference = [
            (sum(sample[0][i][0] for sample in center) / count,
             sum(sample[0][i][1] for sample in center) / count)
            for i in range(core.LANDMARK_COUNT)
        ]
        geometry = core.build_eye_geometry(reference)

        points = []
        for (sx, sy), samples in zip(CALIBRATION_TARGETS, samples_per_target):
            rows = []
            for landmarks, irises in samples:
                sim = core.fit_similarity(reference, landmarks, core.ALIGN_LANDMARKS)
                eye = core.extract_eye_ratios(irises, geometry, sim)
                if eye is None:
                    continue  # 瞳孔が取れていないフレームは使わない
                yaw, pitch, _ = core.head_proxies(landmarks)
                rows.append(core.fill_basis(eye, yaw, pitch, sim.scale))
            if len(rows) < max(4, MIN_SAMPLES // 2):
                self.hint = "瞳孔を検出できない点がありました。明るさを上げるかカメラに近づいて、もう一度 CALIBRATE してください"
                self.model = None
                self.save_model()
                return
            row, _ = robust_row(rows)
            points.append({"row": row, "x": sx, "y": sy})

        fused = core.calibrate(points, (self.width + self.height) / 2.0)
        eye_h = [p["row"][core.F_EYE_H] for p in points]
        eye_v = [p["row"][core.F_EYE_V] for p in points]
        self.model = core.GazeModel.from_calibration(
            fused, reference, geometry,
            {"width": self.width, "height": self.height,
             "eye_range": [[min(eye_h), max(eye_h)], [min(eye_v), max(eye_v)]]},
        )
        self.calib_error = fused["err_px"]
        self.save_model()
        self.reset_filters()
        note = f"（{fused['dropped']}点は外れ値として除外）" if fused["dropped"] else ""
        self.hint = f"キャリブレーション完了 — 誤差 約 {fused['err_px']:.0f} px {note}。t で追跡開始"

    def save_model(self) -> None:
        if self.model is None:
            return
        payload = {
            "version": MODEL_VERSION,
            "mean": self.model.mean, "std": self.model.std,
            "coef_x": self.model.coef_x, "coef_y": self.model.coef_y,
            "err_px": self.model.err_px, "reference": self.model.reference, "meta": self.model.meta,
        }
        try:
            with open(os.path.join(app_home(), "model.json"), "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
        except OSError:
            pass

    def load_model(self) -> None:
        path = os.path.join(app_home(), "model.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("version") != MODEL_VERSION:
                return  # 古い世代（壊れた推定で保存されたもの）は使わない
            reference = [tuple(p) for p in payload["reference"]]
            if len(reference) != core.LANDMARK_COUNT or len(payload["mean"]) != core.FEATURE_COUNT:
                return
            self.model = core.GazeModel(
                payload["mean"], payload["std"], payload["coef_x"], payload["coef_y"],
                payload.get("err_px", 0.0), reference,
                core.build_eye_geometry(reference), payload.get("meta", {}),
            )
            self.calib_error = payload.get("err_px", 0.0)
            self.hint = "前回のキャリブレーションを読み込みました。t で追跡開始"
        except (OSError, KeyError, ValueError, TypeError):
            self.model = None

    # ------------------------------------------------------------------
    # 追跡
    # ------------------------------------------------------------------

    def toggle_tracking(self) -> None:
        if self.controller.state in (core.CursorController.ARMED, core.CursorController.PAUSED):
            self.stop_tracking("停止しました")
            return
        if self.model is None:
            self.hint = "キャリブレーションがまだです。c を押すか CALIBRATE をクリックしてください"
            return
        self.reset_filters()
        self.controller.arm(time.monotonic() * 1000.0, self.width, self.height, position=self.cursor)
        self.hint = "目線でカーソルが動きます。止めたいときは t"

    def stop_tracking(self, reason: str = "") -> None:
        if self.controller.state != core.CursorController.DISARMED:
            self.controller.stop(reason or "stopped")
        if reason:
            self.hint = reason

    def reset_filters(self) -> None:
        self.filters = {
            "pose_yaw": core.OneEuroFilter(3.0, 0.0, 1.0),
            "pose_pitch": core.OneEuroFilter(3.0, 0.0, 1.0),
            "pose_scale": core.OneEuroFilter(3.0, 0.0, 1.0),
            "eye_h": core.OneEuroFilter(12.0, 0.0, 1.0),
            "eye_v": core.OneEuroFilter(12.0, 0.0, 1.0),
            "out_x": core.OneEuroFilter(core.cutoff_from_smoothing(40.0, 0.4, 1.6), 0.012, 1.0),
            "out_y": core.OneEuroFilter(core.cutoff_from_smoothing(40.0, 0.4, 1.6), 0.012, 1.0),
        }
        self.medians = {"x": core.MedianFilter(5), "y": core.MedianFilter(5)}
        self.eye_median.reset()
        self.last_eye = None
        self.rejected = 0
        self.iris_missing = 0
        self.gaze_ever = False

    def estimate(self, observation, now_ms: float) -> bool:
        """観測 → 視線（ウィンドウ内 px）。使えないフレームは False（＝前の値を保持）。"""
        if self.model is None or observation is None:
            return False
        landmarks = observation["landmarks"]
        irises = observation["irises"]

        sim = core.fit_similarity(self.model.reference, landmarks, core.ALIGN_LANDMARKS)
        self.fit_error = sum(
            math.hypot(*(a - b for a, b in zip(sim.apply(*self.model.reference[i]), landmarks[i])))
            for i in core.ALIGN_LANDMARKS
        ) / len(core.ALIGN_LANDMARKS)

        # 瞬き中は虹彩が当てにならないので観測を出さずに保持する
        self.openness = observation["openness"]
        self.eye_median.push(self.openness)
        baseline = self.eye_median.value if len(self.eye_median.buffer) >= 5 else 0.0
        if baseline > 0 and self.openness < baseline * 0.45:
            return False

        eye = core.extract_eye_ratios(irises, self.model.eye_geometry, sim)
        if eye is None:
            self.iris_missing += 1
            return False
        # 検出が別の塊を掴むと h/v が飛ぶ。前回の値から大きく離れたら捨てて保持する。
        if self.last_eye is not None:
            if abs(eye[0] - self.last_eye[0]) > 0.25 or abs(eye[1] - self.last_eye[1]) > 0.25:
                self.rejected += 1
                return False
        self.last_eye = eye
        yaw, pitch, _ = core.head_proxies(landmarks)

        filtered_eye = (self.filters["eye_h"].filter(eye[0], now_ms),
                        self.filters["eye_v"].filter(eye[1], now_ms))
        filtered_pose = (self.filters["pose_yaw"].filter(yaw, now_ms),
                         self.filters["pose_pitch"].filter(pitch, now_ms))
        scale = self.filters["pose_scale"].filter(sim.scale, now_ms)
        self.pose = filtered_pose

        # キャリブ時に見ていた範囲の外を見ていないか（外なら推定は外挿になる）
        eye_range = (self.model.meta or {}).get("eye_range")
        self.frames += 1
        if eye_range:
            inside = (eye_range[0][0] <= filtered_eye[0] <= eye_range[0][1]
                      and eye_range[1][0] <= filtered_eye[1] <= eye_range[1][1])
            if not inside:
                self.out_of_range += 1

        row = core.fill_basis(filtered_eye, filtered_pose[0], filtered_pose[1], scale)
        predicted = self.model.predict(row, self.mode)
        if predicted is None:
            return False

        mx = self.medians["x"].push(predicted[0] * self.width)
        my = self.medians["y"].push(predicted[1] * self.height)
        self.gaze_px = (self.filters["out_x"].filter(mx, now_ms), self.filters["out_y"].filter(my, now_ms))
        self.gaze = (self.gaze_px[0] / self.width, self.gaze_px[1] / self.height)
        self.gaze_ever = True
        return True

    # ------------------------------------------------------------------
    # 定期処理
    # ------------------------------------------------------------------

    def _control_tick(self) -> None:
        if not self.running:
            return
        now_ms = time.monotonic() * 1000.0
        step = self.controller.tick(now_ms)
        if step != (0.0, 0.0):
            self.cursor = (min(max(self.cursor[0] + step[0], 0.0), self.width),
                           min(max(self.cursor[1] + step[1], 0.0), self.height))
        self.controller.set_position(self.cursor)
        if self.controller.click_requested:
            self.controller.click_requested = False
            tile = self._tile_at(self.cursor)
            if tile is not None:
                self.tile_counts[tile] += 1
        self.root.after(10, self._control_tick)

    def _ui_tick(self) -> None:
        if not self.running:
            return
        frame = None
        status = "カメラ停止中"
        fps = 0.0
        if self.worker is not None:
            observation, frame, fps, status, error = self.worker.snapshot()
            now_ms = time.monotonic() * 1000.0
            if error:
                self.hint = error
            if self.calib is not None:
                self._advance_calibration(observation, now_ms)
            elif self.controller.state in (core.CursorController.ARMED, core.CursorController.PAUSED):
                self.estimate(observation, now_ms)
        self._draw(frame)
        self._draw_hud(fps, status)
        self.root.after(16, self._ui_tick)

    def _tile_at(self, point) -> str | None:
        for tile in self.tile_rects:
            x0, y0, x1, y1 = self.canvas.coords(tile["rect"])
            if x0 <= point[0] <= x1 and y0 <= point[1] <= y1:
                return tile["key"]
        return None

    # ------------------------------------------------------------------
    # 描画
    # ------------------------------------------------------------------

    def _draw(self, frame) -> None:
        canvas = self.canvas
        armed = self.controller.state == core.CursorController.ARMED
        hovered = self._tile_at(self.cursor) if armed else None
        for tile in self.tile_rects:
            active = tile["key"] == hovered
            canvas.itemconfig(tile["rect"], outline=ACCENT if active else BORDER,
                              fill="#251d2b" if active else "#1a1520")
            canvas.itemconfig(tile["label"], fill=ACCENT if active else MUTED)
            canvas.itemconfig(tile["count"], text=str(self.tile_counts[tile["key"]]))

        x, y = self.cursor
        canvas.coords(self.canvas_items["cursor_shadow"], x - 14, y - 13, x + 16, y + 17)
        canvas.coords(self.canvas_items["cursor_ring"], x - 13, y - 13, x + 13, y + 13)
        canvas.coords(self.canvas_items["cursor"], x - 6, y - 6, x + 6, y + 6)
        progress = self.controller.dwell_progress if armed else 0.0
        canvas.coords(self.canvas_items["dwell"], x - 18, y - 18, x + 18, y + 18)
        canvas.itemconfig(self.canvas_items["dwell"], extent=-progress * 360,
                          state="normal" if progress > 0 else "hidden")

        show_gaze = self.gaze_ever and self.controller.state in (core.CursorController.ARMED,
                                                                core.CursorController.PAUSED)
        gx, gy = self.gaze_px
        canvas.coords(self.canvas_items["gaze"], gx - 9, gy, gx + 9, gy)
        canvas.coords(self.canvas_items["gaze2"], gx, gy - 9, gx, gy + 9)
        canvas.itemconfig(self.canvas_items["gaze"], state="normal" if show_gaze else "hidden")
        canvas.itemconfig(self.canvas_items["gaze2"], state="normal" if show_gaze else "hidden")

        if self.calib is not None:
            index = self.calib["index"]
            cx, cy = CALIBRATION_TARGETS[index][0] * self.width, CALIBRATION_TARGETS[index][1] * self.height
            canvas.coords(self.canvas_items["calib"], cx - 16, cy - 16, cx + 16, cy + 16)
            canvas.coords(self.canvas_items["calib_dot"], cx - 5, cy - 5, cx + 5, cy + 5)
            canvas.coords(self.canvas_items["calib_ring"], cx - 22, cy - 22, cx + 22, cy + 22)
            gained = len(self.calib["samples"][index])
            canvas.itemconfig(self.canvas_items["calib_ring"],
                              extent=-min(1.0, gained / SAMPLES_PER_TARGET) * 360)
            canvas.coords(self.canvas_items["calib_text"], cx, cy + 44)
            canvas.itemconfig(self.canvas_items["calib_text"],
                              text=f"{index + 1}/{len(CALIBRATION_TARGETS)} 点を見つめてください")
            for name in ("calib", "calib_dot", "calib_ring", "calib_text"):
                canvas.itemconfig(self.canvas_items[name], state="normal")
            canvas.itemconfig(self.canvas_items["hint"], state="hidden")
        else:
            for name in ("calib", "calib_dot", "calib_ring", "calib_text"):
                canvas.itemconfig(self.canvas_items[name], state="hidden")
            canvas.itemconfig(self.canvas_items["hint"], text=self.hint, state="normal")

        self._draw_camera_overlay(frame)

    def _draw_camera_overlay(self, frame) -> None:
        """カメラ映像を画面全体にうっすら重ね、検出した目と瞳孔をくっきり描く。

        Tk の Canvas に透明度は無いので、画像自体を背景色へ寄せて暗くする。
        更新は 10Hz で十分（薄い背景なので滑らかさは要らない）。
        """
        canvas = self.canvas
        observation = self.worker.observation if self.worker is not None else None
        if not self.debug or frame is None:
            canvas.itemconfig(self.canvas_items["camera"], state="hidden")
            for index in range(2):
                canvas.itemconfig(self.canvas_items[f"mark_iris{index}"], state="hidden")
                canvas.itemconfig(self.canvas_items[f"mark_note{index}"], state="hidden")
            return

        import cv2
        import numpy as np

        height, width = frame.shape[:2]
        # 表示は「縮小画像 + Tk の zoom」で作る（毎フレーム巨大な画像を作らないため）
        zoom = 2 if (self.width > 900 or self.height > 600) else 1
        view_w = max(1, int(math.ceil(self.width / zoom)))
        view_h = max(1, int(math.ceil(self.height / zoom)))
        cover = max(view_w / width, view_h / height)
        target = (max(2, int(width * cover)), max(2, int(height * cover)))
        now = time.monotonic()
        if now - self.camera_updated_at >= 0.10:
            self.camera_updated_at = now
            try:
                small = cv2.resize(frame, target, interpolation=cv2.INTER_AREA)
                small = cv2.flip(small, 1)  # 鏡像（見慣れた向き）
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                blended = cv2.addWeighted(gray, 0.20, np.full_like(gray, 22), 0.80, 0)
                ok, buffer = cv2.imencode(".ppm", cv2.cvtColor(blended, cv2.COLOR_GRAY2RGB))
                if ok:
                    photo = tk.PhotoImage(data=buffer.tobytes(), format="ppm")
                    self.photo = photo.zoom(zoom) if zoom > 1 else photo
                    canvas.itemconfig(self.canvas_items["camera"], image=self.photo, state="normal")
                    self.camera_view = (target[0] * zoom, target[1] * zoom)
            except Exception as error:  # 表示は失敗しても本体を止めない
                self.hint = f"カメラ表示に失敗: {error}"
        canvas.itemconfig(self.canvas_items["camera"], state="normal")
        if self.camera_view is None:
            return

        # 検出点を画面座標へ（カバー配置＋鏡像を反映）
        shown_w, shown_h = self.camera_view
        offset_x = (self.width - shown_w) / 2.0
        offset_y = (self.height - shown_h) / 2.0
        scale_x = shown_w / float(width)
        scale_y = shown_h / float(height)

        def to_screen(point):
            return (offset_x + (width - point[0]) * scale_x, offset_y + point[1] * scale_y)

        notes = (observation or {}).get("iris_notes") or ["", ""]
        irises = (observation or {}).get("irises") or [None, None]
        for index in range(2):
            iris = irises[index]
            item = self.canvas_items[f"mark_iris{index}"]
            if iris is None:
                canvas.coords(item, 0, 0, 0, 0)
                canvas.itemconfig(item, state="hidden")
                canvas.coords(self.canvas_items[f"mark_note{index}"], 24, self.height - 120 + index * 16)
                canvas.itemconfig(self.canvas_items[f"mark_note{index}"],
                                  text=f"eye{index + 1} 検出できず", state="normal")
                continue
            x, y = to_screen(iris)
            canvas.coords(item, x - 8, y - 8, x + 8, y + 8)
            canvas.itemconfig(item, state="normal")
            canvas.coords(self.canvas_items[f"mark_note{index}"], 24, self.height - 120 + index * 16)
            canvas.itemconfig(self.canvas_items[f"mark_note{index}"],
                              text=f"eye{index + 1} {notes[index]}", state="normal")

    def _draw_hud(self, fps: float, status: str) -> None:
        state = self.controller.state
        reason = f" ({self.controller.reason})" if self.controller.reason and state != core.CursorController.ARMED else ""
        latency = f"{self.controller.latency_ms:.0f}ms" if self.controller.latency_ms is not None else "--"
        rows = {
            "STATE": f"{state}{reason}",
            "GAZE": f"{self.gaze[0] * 100:.0f}, {self.gaze[1] * 100:.0f}" if self.gaze_ever else "--, --",
            "POSE": f"{self.pose[0]:+.2f}, {self.pose[1]:+.2f}" if self.gaze_ever else "--, --",
            "FIT": f"{self.fit_error:.1f}px" if self.model is not None else "--",
            "RANGE": f"外{100.0 * self.out_of_range / max(self.frames, 1):.0f}%",
            "OPEN": f"{self.openness:.2f} 検出NG{self.iris_missing}",
            "LAT": latency,
            "MODE": dict(MODES)[self.mode],
        }
        for name, text in rows.items():
            self.canvas.itemconfig(self.hud_items[name], text=f"{name} {text}")
        self.canvas.itemconfig(self.canvas_items["status"],
                               text=f"{status} / {fps:.0f}fps / cursor {self.cursor[0]:.0f},{self.cursor[1]:.0f}")
        for button in self.buttons:
            label = button["name"].upper()
            if button["name"] == "mode":
                label = f"MODE {dict(MODES)[self.mode]}"
            elif button["name"] == "track":
                label = "STOP" if state == core.CursorController.ARMED else "TRACKING"
            elif button["name"] == "debug":
                label = "DEBUG ON" if self.debug else "DEBUG"
            self.canvas.itemconfig(button["label"], text=label)


def self_test(camera_index: int) -> int:
    """カメラを開いて、顔と虹彩が取れているかを確認する（GUI なし）。"""
    import gaze_vision

    print(f"カメラ /dev/video{camera_index} を確認します（12 秒ほど）")
    worker = gaze_vision.VisionWorker(camera_index)
    worker.start()
    deadline = time.monotonic() + 12.0
    faces = iris_ok = loops = 0
    while time.monotonic() < deadline:
        observation, _, fps, status, error = worker.snapshot()
        loops += 1
        if error:
            print("エラー:", error)
            worker.stop()
            return 1
        if observation is not None:
            faces += 1
            if all(iris is not None for iris in observation["irises"]):
                iris_ok += 1
        time.sleep(0.1)
    worker.stop()
    print(f"  サンプル {loops} / 顔を検出 {faces} / 虹彩も検出 {iris_ok} / {fps:.1f}fps")
    if faces == 0:
        print("顔が検出できませんでした。写りを確認して明るさ・距離を調整してください")
        return 1
    print("OK: 顔と虹彩が取れています")
    return 0


def robust_row(rows):
    """中央値から大きく外れたフレームを捨てて平均する（瞬き・検出ミス対策）。"""
    medians = [sorted(r[j] for r in rows)[len(rows) // 2] for j in range(core.FEATURE_COUNT)]
    deviations = [sorted(abs(r[j] - medians[j]) for r in rows)[len(rows) // 2] for j in range(core.FEATURE_COUNT)]
    kept = []
    for row in rows:
        okay = True
        for j in range(core.FEATURE_COUNT):
            limit = max(3.0 * deviations[j], 0.05)
            if abs(row[j] - medians[j]) > limit:
                okay = False
                break
        if okay:
            kept.append(row)
    if len(kept) < max(6, len(rows) // 3):
        kept = rows
    return [sum(r[j] for r in kept) / len(kept) for j in range(core.FEATURE_COUNT)], len(rows) - len(kept)


def smoke_test() -> int:
    """カメラを開かずに GUI の組み立てと描画だけ確認する（自動チェック用）。"""
    root = tk.Tk()
    app = GazeApp(root, 0, windowed=True)
    import numpy as np

    fake = np.full((720, 1280, 3), 90, np.uint8)
    app.debug = True
    for _ in range(30):
        app._draw(fake)
        app._draw_hud(0.0, "smoke")
        root.update()
    app.shutdown()
    print("smoke-test: GUI の組み立てと描画は OK（カメラは開いていません）")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="gaze-cursor desktop app（実カーソルは操作しません）")
    parser.add_argument("--camera", type=int, default=0, help="カメラ番号 (/dev/videoN)")
    parser.add_argument("--windowed", action="store_true", help="全画面ではなくウィンドウで起動")
    parser.add_argument("--fetch-model", action="store_true", help="dlib の学習済みモデルを取得して終了")
    parser.add_argument("--self-test", action="store_true", help="カメラと検出の確認だけして終了")
    parser.add_argument("--smoke-test", action="store_true", help="GUI の組み立てだけ確認して終了")
    args = parser.parse_args(argv)

    if args.fetch_model:
        import gaze_vision

        gaze_vision.fetch_model()
        return 0
    if args.self_test:
        return self_test(args.camera)
    if args.smoke_test:
        return smoke_test()

    root = tk.Tk()
    app = GazeApp(root, args.camera, args.windowed)
    root.protocol("WM_DELETE_WINDOW", app.shutdown)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
