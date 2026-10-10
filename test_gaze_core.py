#!/usr/bin/env python3
"""gaze_core の数値テスト（合成データ・ウィンドウ不要）。

    python3 test_gaze_core.py

ブラウザ版 (homepage/others/gaze-cursor) のテストと同じ観点を 2D 版で確認する:
  - 相似変換が既知の変換を復元できるか
  - 頭が動いても眼球特徴が変わらないか（このアプリの核心）
  - 視線 + 頭の向きの統合が、視線のみより良いか
  - 外れ値の除去、制御側の不感帯・注視クリック・フェイルストップ
"""

from __future__ import annotations

import math
import random

import gaze_core as core

PASS: list[str] = []
FAIL: list[str] = []


def ok(condition: bool, message: str) -> None:
    (PASS if condition else FAIL).append(message)
    print(("PASS  " if condition else "FAIL  ") + message)


# ---------------------------------------------------------------------------
# 合成した 68 点の顔（剛体）。座標は「顔の幅 = 2」くらいの適当な単位。
# ---------------------------------------------------------------------------


def make_face() -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = [(0.0, 0.0)] * 68
    # 顎 0-16: 左端 → 顎先 → 右端 の弧
    for i in range(17):
        t = i / 16.0
        points[i] = (-1.0 + 2.0 * t, 0.55 + 0.45 * math.sin(math.pi * t))
    # 眉 17-21(画像左) / 22-26(画像右)
    for i, y in enumerate([-0.50, -0.53, -0.54, -0.53, -0.50]):
        points[17 + i] = (-0.62 + 0.13 * i, y)
        points[22 + i] = (0.62 - 0.13 * (4 - i), y)
    # 鼻 27-35
    points[27] = (0.0, -0.35)
    points[28] = (0.0, -0.25)
    points[29] = (0.0, -0.15)
    points[30] = (0.0, -0.02)  # 鼻先
    points[31] = (-0.12, 0.05)
    points[32] = (-0.06, 0.06)
    points[33] = (0.0, 0.06)   # 鼻の付け根（下）
    points[34] = (0.06, 0.06)
    points[35] = (0.12, 0.05)
    # 目 36-41(画像左) / 42-47(画像右)。両端は X 方向、瞼は Y 方向。
    for base, center in ((36, -0.40), (42, 0.40)):
        points[base + 0] = (center - 0.11, -0.20)
        points[base + 1] = (center - 0.05, -0.26)
        points[base + 2] = (center + 0.05, -0.26)
        points[base + 3] = (center + 0.11, -0.20)
        points[base + 4] = (center + 0.05, -0.14)
        points[base + 5] = (center - 0.05, -0.14)
    # 口 48-67
    for i in range(5):
        points[48 + i] = (-0.30 + 0.15 * i, 0.40)
        points[54 + i] = (0.30 - 0.15 * i, 0.40)
    for i in range(5):
        points[60 + i] = (0.0, 0.40)
    return points


def transform_face(points, scale=1.0, angle_deg=0.0, tx=0.0, ty=0.0):
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    return [(scale * (c * x - s * y) + tx, scale * (s * x + c * y) + ty) for x, y in points]


def iris_for(u: float, v: float, yaw: float = 0.0) -> list[tuple[float, float]]:
    """基準フレームでの眼球の位置 (u, v) から、両目の虹彩の座標を作る。

    yaw は「顔が横を向いた」ことによる見かけのずれ（鼻が横にずれる量を模す）。
    """
    irises = []
    for center in (-0.40, 0.40):
        length = 0.22
        height = 0.12
        irises.append((center - length / 2 + length * u + yaw * 0.3, -0.26 + height * v - 0.03))
    return irises


BASE_FACE = make_face()
IDENTITY = core.Similarity(1.0, 1.0, 0.0, 0.0, 0.0)

print("=== 相似変換 ===")
truth = core.Similarity(1.3, math.cos(math.radians(17)), math.sin(math.radians(17)), 0.4, -0.2)
observed = [truth.apply(x, y) for x, y in BASE_FACE]
fitted = core.fit_similarity(BASE_FACE, observed, core.ALIGN_LANDMARKS)
residual = max(
    math.hypot(*(a - b for a, b in zip(fitted.apply(*BASE_FACE[i]), observed[i])))
    for i in core.ALIGN_LANDMARKS
)
ok(abs(fitted.scale - 1.3) < 1e-9 and residual < 1e-9,
   f"既知の相似変換を復元 (scale={fitted.scale:.4f} 残差={residual:.2e})")
back = fitted.inverse(*truth.apply(1.0, 2.0))
ok(abs(back[0] - 1.0) < 1e-9 and abs(back[1] - 2.0) < 1e-9, "逆変換で元に戻る")

print()
print("=== 頭が動いても眼球特徴が変わらないか（核心） ===")
geometry = core.build_eye_geometry(BASE_FACE)
target = (0.63, 0.42)
worst = 0.0
cases = [
    ("そのまま", dict()),
    ("平行移動", dict(tx=0.4, ty=-0.3)),
    ("距離が違う", dict(scale=1.35)),
    ("傾き(roll)", dict(angle_deg=12)),
    ("全部", dict(scale=0.8, angle_deg=-9, tx=-0.2, ty=0.25)),
]
for name, kwargs in cases:
    observed_face = transform_face(BASE_FACE, **kwargs)
    observed_irises = [transform_face([p], **kwargs)[0] for p in iris_for(*target)]
    sim = core.fit_similarity(BASE_FACE, observed_face, core.ALIGN_LANDMARKS)
    eye = core.extract_eye_ratios(observed_irises, geometry, sim)
    worst = max(worst, abs(eye[0] - target[0]), abs(eye[1] - target[1]))
    print(f"       {name:10s} → h={eye[0]:.4f} v={eye[1]:.4f} (真値 {target[0]}, {target[1]})")
ok(worst < 0.02, f"頭の平行移動・距離・傾きで眼球特徴が不変（最大ずれ {worst:.4f}）")

# 参考: 位置合わせをしない（画像座標のまま）と、頭が動くと特徴が動いてしまう
raw = []
for name, kwargs in [("そのまま", dict()), ("平行移動", dict(tx=0.4, ty=-0.3))]:
    observed_irises = [transform_face([p], **kwargs)[0] for p in iris_for(*target)]
    geometry_raw = core.build_eye_geometry(BASE_FACE)
    eye_raw = core.extract_eye_ratios(observed_irises, geometry_raw, IDENTITY)
    raw.append(eye_raw[0])
ok(abs(raw[0] - raw[1]) > 0.05,
   f"補正しないと平行移動だけで特徴がずれる（{raw[0]:.3f} → {raw[1]:.3f}）")

print()
print("=== 頭の向きの代理量 ===")
# 鼻が横にずれた顔（＝横を向いた）を作る
yawed = list(BASE_FACE)
for i in (27, 28, 29, 30, 31, 32, 33, 34, 35):
    yawed[i] = (yawed[i][0] + 0.18, yawed[i][1])
base_yaw, base_pitch, dist = core.head_proxies(BASE_FACE)
yaw_yaw, _, _ = core.head_proxies(yawed)
moved_yaw, _, _ = core.head_proxies(transform_face(BASE_FACE, scale=1.4, tx=0.5))
ok(abs(yaw_yaw - base_yaw) > 0.05, f"鼻がずれると yaw 代理量が変わる（{base_yaw:.3f} → {yaw_yaw:.3f}）")
ok(abs(moved_yaw - base_yaw) < 1e-9, f"拡大・平行移動では変わらない（{moved_yaw:.3f}）")

print()
print("=== キャリブレーションと統合 ===")
random.seed(7)
geometry = core.build_eye_geometry(BASE_FACE)
# 真実: 画面の点を見るには、頭を 40% 回し、残りを眼球でカバーする
def truth(sx, sy):
    angle_x = (sx - 0.5) * 2.0
    angle_y = (sy - 0.5) * 2.0
    yaw = 0.4 * angle_x * 0.9
    pitch = -0.4 * angle_y * 0.7
    u = 0.5 + (angle_x * 0.9 - yaw) * 0.35
    v = 0.5 - (angle_y * 0.7 - pitch) * 0.35
    return yaw, pitch, u, v

def row_for(sx, sy, noise=0.0):
    yaw, pitch, u, v = truth(sx, sy)
    j = lambda: random.uniform(-noise, noise)
    return core.fill_basis((u + j(), v + j()), yaw + j(), pitch + j(), 1.0 + j())

grid = [0.1, 0.3, 0.5, 0.7, 0.9]
cal = [(0.5, 0.5), (0.1, 0.1), (0.5, 0.1), (0.9, 0.1), (0.1, 0.5), (0.9, 0.5),
       (0.1, 0.9), (0.5, 0.9), (0.9, 0.9), (0.3, 0.1), (0.7, 0.1), (0.3, 0.9), (0.7, 0.9)]
points = [{"row": row_for(sx, sy, 0.004), "x": sx, "y": sy} for sx, sy in cal]
fitted = core.calibrate(points, 700.0)
model = core.GazeModel.from_calibration(fitted, BASE_FACE, geometry)

def evaluate(mode):
    total = 0.0
    worst = 0.0
    n = 0
    for sx in grid:
        for sy in grid:
            got = model.predict(row_for(sx, sy), mode)
            err = math.hypot(got[0] - sx, got[1] - sy) * 700.0
            total += err
            worst = max(worst, err)
            n += 1
    return total / n, worst

both, worst = evaluate(core.MODE_BOTH)
gaze_only, _ = evaluate(core.MODE_GAZE)
head_only, _ = evaluate(core.MODE_HEAD)
print(f"       未学習25点の平均誤差: 視線+頭={both:.1f}px 視線のみ={gaze_only:.1f}px 頭のみ={head_only:.1f}px"
      f" (最悪 {worst:.1f}px, in-sample {fitted['err_px']:.1f}px, 除外 {fitted['dropped']}点)")
ok(both < 30, f"統合で誤差が小さい（{both:.1f}px < 30px）")
ok(both < gaze_only * 0.8, f"頭の向きを入れると視線のみより良い（{both:.1f} < {gaze_only * 0.8:.1f}）")

dirty = [dict(p) for p in points]
dirty[5] = {**dirty[5], "x": 0.1, "y": 0.9}
robust = core.calibrate(dirty, 700.0)
ok(robust["dropped"] == 1, f"外れ値1点を除外する（dropped={robust['dropped']}）")

print()
print("=== フィルタ ===")
median = core.MedianFilter(5)
values = []
for i in range(60):
    values.append(median.push(500.0 if i % 2 == 0 else 520.0))
spread = max(values[20:]) - min(values[20:])
ok(spread < 30.0, f"中央値5で交互の振動を落とす（幅 {spread:.1f}px）")

euro = core.OneEuroFilter(1.0, 0.0, 1.0)
ramp = [euro.filter(100.0 * (i / 60.0), i * (1000.0 / 60.0)) for i in range(61)]
lag = 100.0 - ramp[-1]
ok(5.0 < lag < 40.0, f"1€/EMA の遅れが妥当（{lag:.1f}px）")

print()
print("=== 制御（不感帯・注視クリック・フェイルストップ） ===")
controller = core.CursorController({"deadzone_px": 40.0, "pause_after_ms": 400.0})
now = 10000.0
controller.arm(now, 1000.0, 600.0)
controller.set_position((0.0, 0.0))
step = controller.tick(now)
ok(step == (0.0, 0.0) and controller.state == core.CursorController.PAUSED,
   f"観測が来るまで動かない（{controller.state}）")

now += 10
controller.observe_gaze(now, now, 0.5, 0.5)
travel = 0.0
for _ in range(20):
    now += 10
    controller.observe_gaze(now, now - 5, 0.5, 0.5)
    step = controller.tick(now)
    controller.set_position((controller.pos[0] + step[0], controller.pos[1] + step[1]))
    travel += math.hypot(*step)
gap = math.hypot(controller.pos[0] - 500.0, controller.pos[1] - 300.0)
ok(gap <= 41.0, f"不感帯(40px)の内側まで寄る（残り {gap:.0f}px）")

# 推定が毎周期 画面の端と端を往復しても、不感帯のおかげでカーソルはほとんど動かない
before = controller.moved_px
for i in range(30):
    now += 10
    controller.observe_gaze(now, now, 1.0 if i % 2 else 0.0, 1.0 if i % 2 else 0.0)
    step = controller.tick(now)
    controller.set_position((controller.pos[0] + step[0], controller.pos[1] + step[1]))
extra = controller.moved_px - before
ok(extra < 200.0, f"荒ぶる推定を追わない（0.3秒の総移動 {extra:.0f}px）")

# 注視クリック
controller2 = core.CursorController({"deadzone_px": 40.0})
now = 10000.0
controller2.arm(now, 1000.0, 600.0, position=(500.0, 300.0))
now += 10
controller2.observe_gaze(now, now, 0.5, 0.5)
controller2.click_requested = False
click_at = None
for _ in range(300):
    now += 10
    controller2.observe_gaze(now, now - 5, 0.5, 0.5)
    controller2.tick(now)
    if controller2.click_requested:
        click_at = now - 10000.0
        break
ok(click_at is not None and 880 <= click_at <= 1100,
   f"注視 900ms でクリック（{click_at}ms, 回数 {controller2.clicks}）")

# フェイルストップ
controller3 = core.CursorController({"pause_after_ms": 400.0, "fail_stop_after_ms": 2500.0})
now = 10000.0
controller3.arm(now, 1000.0, 600.0)
now += 10
controller3.observe_gaze(now, now, 0.5, 0.5)
controller3.tick(now)
now += 500
controller3.tick(now)
ok(controller3.state == core.CursorController.PAUSED, "400ms 途切れたら一時停止")
now += 2200
controller3.tick(now)
ok(controller3.state == core.CursorController.STOPPED, "2.5s 途切れたら要再アーム")

print()
print(f"--- PASS {len(PASS)} / FAIL {len(FAIL)} ---")
raise SystemExit(1 if FAIL else 0)
