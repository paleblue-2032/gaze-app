"""カメラと顔・虹彩の検出（プラットフォーム層）。

ここだけが numpy / opencv / dlib に依存する。数値計算は gaze_core.py 側。

流れ:
    カメラ → dlib で顔矩形 → 68点ランドマーク → 目ごとに瞳孔/虹彩の重心を画像処理で求める
          → 「観測」（ランドマーク・虹彩・目の開き・時刻）を作ってアプリに渡す

顔検出は毎フレームやらず、数フレームに一度だけ検出して、間は前回の矩形を使う
（shape_predictor は矩形さえあれば速い）。CPU でも 20〜30fps 出る。
"""

from __future__ import annotations

import math
import os
import threading
import time

import cv2
import dlib
import numpy as np

import gaze_core as core

MODEL_URL = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
MODEL_FILE = "shape_predictor_68_face_landmarks.dat"


def model_path() -> str:
    """学習済みモデルの場所。無ければ案内を出す（アプリ側で取得できる）。"""
    home = os.environ.get("GAZE_APP_HOME", os.path.expanduser("~/.cache/gaze-app"))
    for candidate in (os.path.join(home, MODEL_FILE), os.path.join(os.path.dirname(__file__), MODEL_FILE)):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"{MODEL_FILE} が見つかりません。\n"
        f"  {home}/ に置くか、次のコマンドで取得してください:\n"
        f"  curl -sSL {MODEL_URL} | bzip2 -d > {home}/{MODEL_FILE}\n"
        f"（アプリの --fetch-model でも取得できます）"
    )


def fetch_model() -> str:
    """学習済みモデルを取得する（初回のみ）。"""
    import bz2
    import urllib.request

    home = os.environ.get("GAZE_APP_HOME", os.path.expanduser("~/.cache/gaze-app"))
    os.makedirs(home, exist_ok=True)
    target = os.path.join(home, MODEL_FILE)
    print(f"モデルを取得します（約100MB）: {MODEL_URL}")
    with urllib.request.urlopen(MODEL_URL) as response:
        data = bz2.decompress(response.read())
    with open(target, "wb") as handle:
        handle.write(data)
    print(f"保存しました: {target}")
    return target


class EyeRegion:
    """画像処理で瞳孔（無ければ虹彩）の重心を出す。

    ポイントは **目を切り出して拡大してから処理すること**。実画像では目が 30〜40px しかなく、
    そのまま閾値処理すると睫毛や陰影と同じスケールになって破綻する（実際そうなっていた）。
    拡大してから黒帽（black-hat）で暗い塊を強調し、輪郭を円らしさ・大きさ・中央らしさで採点する。
    """

    SHRINK = 0.18           # 目の中をこの割合だけ縮めてから探す（睫毛・瞼の影を避ける）
    UPSCALE_TARGET = 180    # 目の幅がこれくらいになるまで拡大してから処理する
    MIN_SIZE_RATIO = 0.08   # 瞳孔/虹彩の直径 ÷ 目の幅 の下限
    MAX_SIZE_RATIO = 0.95

    @staticmethod
    def region(gray, landmarks, slot):
        """目の多角形（縮めたもの）と切り出し範囲を返す。"""
        upper = min(slot["upper_candidates"], key=lambda i: landmarks[i][1])
        lower = max(slot["lower_candidates"], key=lambda i: landmarks[i][1])
        a, b = slot["corners"]
        polygon = np.array([landmarks[a], landmarks[b], landmarks[lower], landmarks[upper]],
                           dtype=np.float32)
        center = polygon.mean(axis=0)
        polygon = center + (polygon - center) * (1.0 - EyeRegion.SHRINK)
        x0 = int(max(0, math.floor(polygon[:, 0].min())))
        x1 = int(min(gray.shape[1] - 1, math.ceil(polygon[:, 0].max())))
        y0 = int(max(0, math.floor(polygon[:, 1].min())))
        y1 = int(min(gray.shape[0] - 1, math.ceil(polygon[:, 1].max())))
        return polygon, (x0, y0, x1, y1)

    @staticmethod
    def locate(gray: np.ndarray, landmarks, slot, want_debug: bool = False, prior=None):
        polygon, (x0, y0, x1, y1) = EyeRegion.region(gray, landmarks, slot)
        width, height = x1 - x0 + 1, y1 - y0 + 1
        if width < 6 or height < 4:
            return (None, "小さすぎ") if want_debug else None

        scale = min(max(EyeRegion.UPSCALE_TARGET / float(width), 1.0), 8.0)
        roi = gray[y0 : y1 + 1, x0 : x1 + 1]
        if scale > 1.05:
            roi = cv2.resize(roi, (int(width * scale), int(height * scale)),
                             interpolation=cv2.INTER_CUBIC)

        mask = np.zeros(roi.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [np.round((polygon - (x0, y0)) * scale).astype(np.int32)], 255)
        if int((mask > 0).sum()) < 60:
            return (None, "狭すぎ") if want_debug else None

        # 明るさを揃えてから暗い塊を強調
        equalized = cv2.equalizeHist(roi)
        kernel_size = max(5, int(roi.shape[1] * 0.25)) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, max(3, kernel_size // 3)))
        blackhat = cv2.morphologyEx(equalized, cv2.MORPH_BLACKHAT, kernel)
        blackhat[mask == 0] = 0

        # Otsu で「暗い塊」と「それ以外」を分ける
        values = blackhat[mask > 0]
        level, _ = cv2.threshold(values.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        level = max(float(level), float(np.quantile(values, 0.60)), 6.0)
        _, binary = cv2.threshold(blackhat, level, 255, cv2.THRESH_BINARY)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        eye_width = float(roi.shape[1])
        center_x, center_y = eye_width / 2.0, roi.shape[0] / 2.0
        minimum_area = max(4.0, 0.006 * float((mask > 0).sum()))

        def pick(strict_size: bool):
            """暗い塊を採点して選ぶ。strict_size=False なら大きさ条件を外す。"""
            found = None
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < minimum_area:
                    continue
                perimeter = cv2.arcLength(contour, True)
                if perimeter <= 0:
                    continue
                circularity = 4.0 * math.pi * area / (perimeter * perimeter)
                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                ratio = (2.0 * radius) / eye_width
                if strict_size and not (EyeRegion.MIN_SIZE_RATIO <= ratio <= EyeRegion.MAX_SIZE_RATIO):
                    continue
                offset = math.hypot(cx - center_x, cy - center_y) / (eye_width / 2.0)
                score = circularity * min(area / (minimum_area * 3.0), 1.0) * max(0.2, 1.0 - 0.6 * offset)
                if prior is not None:
                    # 前フレームの瞳孔の近くを強く優遇する（睫毛の影などへ乗り換えるのを防ぐ）
                    prior_x = (prior[0] - x0) * scale
                    prior_y = (prior[1] - y0) * scale
                    near = math.hypot(cx - prior_x, cy - prior_y) / (eye_width * 0.6)
                    score *= 1.0 / (1.0 + 2.5 * near * near)
                if found is None or score > found[0]:
                    found = (score, cx, cy, area, circularity, ratio)
            return found

        # きつい条件で探し、見つからなければ条件を緩めて再挑戦する（遠い顔・小さい目でも拾う）
        best = pick(True)
        strict = best is not None
        if best is None:
            best = pick(False)

        if best is None:
            # 円形の塊が取れないときは「暗い方から15%」の重心で代用する。
            # 何も返さないより遥かにマシで、アプリ側の跳び拒否が効く。
            pixels = equalized[mask > 0]
            if pixels.size == 0:
                return (None, "塊なし") if want_debug else None
            cutoff = np.quantile(pixels, 0.15)
            dark = (equalized <= cutoff) & (mask > 0)
            ys, xs = np.nonzero(dark)
            if xs.size < 3:
                return (None, f"塊なし (level={level:.0f})") if want_debug else None
            weights = np.clip(cutoff - equalized, 1.0, None) * dark
            total = weights.sum()
            point = (float((xs * weights[dark]).sum() / total) / scale + x0,
                     float((ys * weights[dark]).sum() / total) / scale + y0)
            return (point, "最暗部で代用") if want_debug else point
        point = (best[1] / scale + x0, best[2] / scale + y0)
        note = f"{'○' if strict else '△緩'} 倍率{scale:.1f} 面積{best[3]:.0f} 円らしさ{best[4]:.2f} 縦横{best[5]:.2f}"
        return (point, note) if want_debug else point


class VisionWorker(threading.Thread):
    """カメラを読み続けて「観測」を作るスレッド。"""

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720, detect_every: int = 12):
        super().__init__(daemon=True, name="gaze-vision")
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.detect_every = detect_every
        self.running = threading.Event()
        self.running.set()

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_path())

        self.lock = threading.Lock()
        self.observation: dict | None = None   # 最新の観測
        self.frame = None                       # デバッグ表示用の最新フレーム
        self.fps = 0.0
        self.error: str | None = None
        self.status = "起動中"
        self._stop_requested = False
        self.camera = None
        self._last_iris = None   # 前フレームの瞳孔位置（検出の手がかり）

    # --- ライフサイクル ---

    def stop(self) -> None:
        self.running.clear()
        self._stop_requested = True

    def describe_camera(self) -> str:
        return f"/dev/video{self.camera_index}"

    def snapshot(self) -> tuple[dict | None, object, float, str, str | None]:
        with self.lock:
            return self.observation, self.frame, self.fps, self.status, self.error

    # --- 1フレームの処理（画像でも同じ経路を使えるように切り出してある） ---

    def detect(self, frame):
        """顔矩形を返す（無ければ None）。"""
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        faces = self.detector(gray, 0)
        if not faces:
            return None
        return max(faces, key=lambda rect: rect.width() * rect.height())

    def process_frame(self, frame, face=None) -> dict | None:
        """フレーム → 観測（ランドマーク・虹彩・目の開き）。"""
        if face is None:
            face = self.detect(frame)
        if face is None:
            return None
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        shape = self.predictor(gray, face)
        landmarks = [(float(shape.part(i).x), float(shape.part(i).y)) for i in range(core.LANDMARK_COUNT)]
        prior = self._last_iris or (None, None)
        located = [EyeRegion.locate(gray, landmarks, slot, want_debug=True, prior=prior[i])
                   for i, slot in enumerate(core.EYE_SLOTS)]
        irises = [item[0] for item in located]
        notes = [item[1] for item in located]
        self._last_iris = irises if all(iris is not None for iris in irises) else None
        return {
            "t": time.monotonic() * 1000.0,
            "landmarks": landmarks,
            "irises": irises,
            "openness": core.eye_openness(landmarks),
            "iris_notes": notes,
            "face": (face.left(), face.top(), face.right(), face.bottom()),
        }

    # --- 本体 ---

    def run(self) -> None:
        self.camera = cv2.VideoCapture(self.camera_index)
        if not self.camera.isOpened():
            with self.lock:
                self.error = f"カメラ /dev/video{self.camera_index} を開けませんでした"
                self.status = "カメラなし"
            return
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        frames = 0
        last_detect = -1
        face = None
        started = time.monotonic()
        while self.running.is_set():
            ok, frame = self.camera.read()
            if not ok:
                time.sleep(0.02)
                continue
            frames += 1
            # 顔検出は間引く。間は前回の矩形を使う（速い）
            if face is None or frames - last_detect >= self.detect_every:
                last_detect = frames
                face = self.detect(frame)
            elif (frames - last_detect) > self.detect_every * 6:
                face = None

            observation = self.process_frame(frame, face)

            elapsed = time.monotonic() - started
            with self.lock:
                self.frame = frame
                if observation is not None:
                    self.observation = observation
                self.fps = frames / elapsed if elapsed > 0 else 0.0
                self.status = "追跡中" if observation is not None else "顔を探しています"
                self.error = None

        if self.camera is not None:
            self.camera.release()


def draw_debug(frame, observation) -> None:
    """デバッグ表示: 検出した目の輪郭と虹彩の位置を描く。"""
    if observation is None:
        return
    landmarks = observation["landmarks"]
    for slot in core.EYE_SLOTS:
        color = (80, 200, 220)
        for index in list(slot["corners"]) + list(slot["upper_candidates"]) + list(slot["lower_candidates"]):
            x, y = landmarks[index]
            cv2.circle(frame, (int(x), int(y)), 2, color, -1)
    notes = observation.get("iris_notes") or ["", ""]
    for index, (iris, slot) in enumerate(zip(observation["irises"], core.EYE_SLOTS)):
        if iris is None:
            cv2.putText(frame, f"eye{index + 1} NG", (8, 160 + index * 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 240), 1)
            continue
        cv2.circle(frame, (int(iris[0]), int(iris[1])), 6, (60, 60, 245), 2)
        cv2.drawMarker(frame, (int(iris[0]), int(iris[1])), (60, 60, 245), cv2.MARKER_CROSS, 11, 2)
        cv2.putText(frame, f"eye{index + 1} {notes[index]}", (8, 160 + index * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 220, 80), 1)
    face = observation.get("face")
    if face:
        cv2.rectangle(frame, (face[0], face[1]), (face[2], face[3]), (60, 160, 120), 1)
