"""gaze-app の推定コア（依存ゼロの純 Python）。

ブラウザ版 (homepage/others/gaze-cursor/core.js) と同じ考え方を 2D で実装したもの:

  1. 顔のランドマーク（dlib 68点）から「基準フレーム」への相似変換（拡大・回転・平行移動）を解く
  2. 虹彩の位置をその変換で基準フレームへ写してから比率を取る
     → 眼球特徴が頭の平行移動・距離変化・傾き（roll）に依存しない
  3. 頭の向きの代理量（鼻の左右/上下のずれ）を特徴量に混ぜて回帰する
  4. 瞬き中は観測を出さずに前の値を保持する
  5. 制御側は不感帯を持ち、推定の細かい揺れを追わない

numpy も opencv も使わない（点は数十個、特徴量は8次元なので十分速い）。
カメラやランドマーク検出は gaze_vision.py 側の仕事で、ここは数値計算だけ。
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# 使うランドマーク（dlib 68点の添字）
# ---------------------------------------------------------------------------

# 位置合わせに使う点。顔全体に散っていて、表情で大きく動かないものを選んである。
# 鼻先(30)・鼻の付け根(33) は頭を振ると 2D で大きくずれるので、位置合わせからは外して
# 「頭の向きの代理量」として別に使う（そちらの方が情報として素直）。
ALIGN_LANDMARKS = [
    0, 8, 16,          # 顎（左端・先端・右端）
    17, 21, 22, 26,    # 眉（左右の端）
    27,                # 眉間
    36, 39, 42, 45,    # 目の両端（左右）
]
# 頭の向きの代理量に使う点
NOSE_TIP = 30
NOSE_BASE = 33
CHIN = 8
BROW_MID = 27
EYE_CORNERS = [36, 39, 42, 45]

# 目（dlib の輪郭6点のうち両端と、上下の瞼の候補）
EYE_SLOTS = [
    {"corners": (36, 39), "upper_candidates": (37, 38), "lower_candidates": (40, 41)},
    {"corners": (42, 45), "upper_candidates": (43, 44), "lower_candidates": (46, 47)},
]

LANDMARK_COUNT = 68

FEATURE_COUNT = 8
F_BIAS = 0
F_EYE_H = 1
F_EYE_V = 2
F_EYE_H2 = 3
F_EYE_V2 = 4
F_YAW = 5
F_PITCH = 6
F_SCALE = 7

MODE_BOTH = "both"
MODE_GAZE = "gaze"
MODE_HEAD = "head"


def clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


# ---------------------------------------------------------------------------
# 2D の相似変換（拡大・回転・平行移動）
# ---------------------------------------------------------------------------


class Similarity:
    """基準フレーム → 観測 の相似変換。scale, cos, sin, tx, ty で表す。"""

    __slots__ = ("scale", "cos", "sin", "tx", "ty")

    def __init__(self, scale: float, cos: float, sin: float, tx: float, ty: float) -> None:
        self.scale = scale
        self.cos = cos
        self.sin = sin
        self.tx = tx
        self.ty = ty

    def apply(self, x: float, y: float) -> tuple[float, float]:
        return (
            self.scale * (self.cos * x - self.sin * y) + self.tx,
            self.scale * (self.sin * x + self.cos * y) + self.ty,
        )

    def inverse(self, x: float, y: float) -> tuple[float, float]:
        """観測 → 基準フレーム。"""
        inv = 1.0 / self.scale if self.scale > 1e-9 else 1.0
        dx, dy = (x - self.tx) * inv, (y - self.ty) * inv
        return (self.cos * dx + self.sin * dy, -self.sin * dx + self.cos * dy)


def fit_similarity(reference, observed, indices) -> Similarity:
    """最小二乗で相似変換を解く（閉形式なので反復が要らない）。"""
    count = len(indices)
    rx = ry = ox = oy = 0.0
    for index in indices:
        rx += reference[index][0]
        ry += reference[index][1]
        ox += observed[index][0]
        oy += observed[index][1]
    rx /= count
    ry /= count
    ox /= count
    oy /= count

    a = b = norm = 0.0
    for index in indices:
        dx = reference[index][0] - rx
        dy = reference[index][1] - ry
        ex = observed[index][0] - ox
        ey = observed[index][1] - oy
        a += dx * ex + dy * ey
        b += dx * ey - dy * ex
        norm += dx * dx + dy * dy
    if norm < 1e-12:
        return Similarity(1.0, 1.0, 0.0, ox - rx, oy - ry)

    scale = math.hypot(a, b) / norm
    cos = a / math.hypot(a, b) if math.hypot(a, b) > 1e-12 else 1.0
    sin = b / math.hypot(a, b) if math.hypot(a, b) > 1e-12 else 0.0
    angle_cos = cos
    angle_sin = sin
    tx = ox - scale * (angle_cos * rx - angle_sin * ry)
    ty = oy - scale * (angle_sin * rx + angle_cos * ry)
    return Similarity(scale, angle_cos, angle_sin, tx, ty)


def project_ratio(px, py, ax, ay, bx, by) -> float:
    """点 p を線分 a→b に射影した比。a で 0、b で 1。"""
    vx, vy = bx - ax, by - ay
    length2 = vx * vx + vy * vy
    if length2 < 1e-12:
        return 0.5
    return ((px - ax) * vx + (py - ay) * vy) / length2


# ---------------------------------------------------------------------------
# 特徴量
# ---------------------------------------------------------------------------


class EyeGeometry:
    """基準フレームでの目の形（両端と上下の瞼）。"""

    def __init__(self, corner_a, corner_b, upper, lower) -> None:
        self.corner_a = corner_a
        self.corner_b = corner_b
        self.upper = upper
        self.lower = lower


def build_eye_geometry(reference) -> list[EyeGeometry]:
    """基準フレーム（キャリブ時の正面姿勢）から目の形を決める。"""
    geometries = []
    for slot in EYE_SLOTS:
        a, b = slot["corners"]
        upper = min(slot["upper_candidates"], key=lambda i: reference[i][1])
        lower = max(slot["lower_candidates"], key=lambda i: reference[i][1])
        geometries.append(EyeGeometry(reference[a], reference[b], reference[upper], reference[lower]))
    return geometries


def extract_eye_ratios(irises, geometries, sim: Similarity):
    """虹彩（画像座標）を基準フレームへ写して、目の形に対する比率を出す。

    虹彩が検出できなかったときは None を返す。以前は (0.5, 0.5) で埋めていたが、
    それは「中央を見ている」という嘘の値で、キャリブレーションを壊し、
    推定を狂わせる（実際にこれで壊れていた）。
    """
    h_sum = v_sum = 0.0
    for iris, geometry in zip(irises, geometries):
        if iris is None:
            return None
        x, y = sim.inverse(iris[0], iris[1])
        h_sum += project_ratio(x, y, geometry.corner_a[0], geometry.corner_a[1],
                               geometry.corner_b[0], geometry.corner_b[1])
        v_sum += project_ratio(x, y, geometry.upper[0], geometry.upper[1],
                               geometry.lower[0], geometry.lower[1])
    return (h_sum / len(geometries), v_sum / len(geometries))


def head_proxies(landmarks) -> tuple[float, float, float]:
    """頭の向きの代理量。

    yaw: 鼻先が両目の中点からどれだけ左右にずれているか（両目間距離で正規化）
    pitch: 顎先が眉間からどれだけ下にあるか（同）
    どちらも縮尺に依存しないので、距離が変わっても値が保たれる。
    """
    left, right = landmarks[EYE_CORNERS[0]], landmarks[EYE_CORNERS[3]]
    eye_mid_x = (landmarks[36][0] + landmarks[39][0] + landmarks[42][0] + landmarks[45][0]) / 4.0
    eye_mid_y = (landmarks[36][1] + landmarks[39][1] + landmarks[42][1] + landmarks[45][1]) / 4.0
    eye_distance = math.hypot(right[0] - left[0], right[1] - left[1])
    if eye_distance < 1e-6:
        eye_distance = 1.0

    yaw = (landmarks[NOSE_TIP][0] - eye_mid_x) / eye_distance
    pitch = (landmarks[CHIN][1] - landmarks[BROW_MID][1]) / eye_distance
    return (yaw, pitch, eye_distance)


def eye_openness(landmarks) -> float:
    """上下の瞼の開き / 目の幅。瞬き判定に使う。"""
    values = []
    for slot in EYE_SLOTS:
        a, b = slot["corners"]
        upper = min(slot["upper_candidates"], key=lambda i: landmarks[i][1])
        lower = max(slot["lower_candidates"], key=lambda i: landmarks[i][1])
        width = math.hypot(landmarks[b][0] - landmarks[a][0], landmarks[b][1] - landmarks[a][1])
        if width > 1e-6:
            height = math.hypot(landmarks[lower][0] - landmarks[upper][0],
                                landmarks[lower][1] - landmarks[upper][1])
            values.append(height / width)
    return sum(values) / len(values) if values else 0.0


def fill_basis(eye, yaw, pitch, scale) -> list[float]:
    h, v = eye
    return [1.0, h, v, h * h, v * v, yaw, pitch, scale]


# ---------------------------------------------------------------------------
# フィルタ
# ---------------------------------------------------------------------------


def smoothing_factor(dt_sec: float, cutoff_hz: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt_sec)


class OneEuroFilter:
    """1€ filter。速い動きでは遅れを減らし、止まっているときは震えを消す。"""

    def __init__(self, min_cutoff_hz: float, beta: float = 0.0, d_cutoff_hz: float = 1.0) -> None:
        self.min_cutoff_hz = min_cutoff_hz
        self.beta = beta
        self.d_cutoff_hz = d_cutoff_hz
        self.reset()

    def reset(self) -> None:
        self.value = 0.0
        self.rate = 0.0
        self.time_ms = 0.0
        self.started = False

    def filter(self, next_value: float, time_ms: float) -> float:
        if not self.started:
            self.value = next_value
            self.time_ms = time_ms
            self.started = True
            return next_value
        dt_sec = (time_ms - self.time_ms) / 1000.0
        dt_sec = min(max(dt_sec, 1e-3), 0.1)
        self.time_ms = time_ms
        rate = (next_value - self.value) / dt_sec
        self.rate += smoothing_factor(dt_sec, self.d_cutoff_hz) * (rate - self.rate)
        cutoff = self.min_cutoff_hz + self.beta * abs(self.rate)
        self.value += smoothing_factor(dt_sec, cutoff) * (next_value - self.value)
        return self.value


class MedianFilter:
    """直近 size 個の中央値。振動と単発スパイクを落とす。"""

    def __init__(self, size: int) -> None:
        self.size = size
        self.buffer: list[float] = []
        self.value = 0.0

    def reset(self) -> None:
        self.buffer.clear()
        self.value = 0.0

    def push(self, value: float) -> float:
        self.buffer.append(value)
        if len(self.buffer) > self.size:
            self.buffer.pop(0)
        ordered = sorted(self.buffer)
        n = len(ordered)
        self.value = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0
        return self.value


def cutoff_from_smoothing(slider: float, slow_hz: float, fast_hz: float) -> float:
    return fast_hz + (slow_hz - fast_hz) * clamp01(slider / 100.0)


# ---------------------------------------------------------------------------
# キャリブレーション（リッジ回帰 + leave-one-out 外れ値除去）
# ---------------------------------------------------------------------------


def make_scaler(rows) -> tuple[list[float], list[float]]:
    count = len(rows)
    mean = [0.0] * FEATURE_COUNT
    std = [1.0] * FEATURE_COUNT
    for j in range(1, FEATURE_COUNT):
        mean[j] = sum(row[j] for row in rows) / count
        variance = sum((row[j] - mean[j]) ** 2 for row in rows) / count
        std[j] = math.sqrt(variance) or 1.0
    return mean, std


def apply_scaler(mean, std, row) -> list[float]:
    return [1.0] + [(row[j] - mean[j]) / std[j] for j in range(1, FEATURE_COUNT)]


def solve_ridge(rows, targets, lam: float) -> list[float]:
    """正規方程式 + ガウス・ジョルダン。ハンドパスで解く。"""
    m = FEATURE_COUNT
    mat = [[0.0] * (m + 1) for _ in range(m)]
    for row, target in zip(rows, targets):
        for r in range(m):
            rv = row[r]
            for c in range(m):
                mat[r][c] += rv * row[c]
            mat[r][m] += rv * target
    trace = sum(mat[i][i] for i in range(m)) / m
    penalty = lam * (trace if trace > 1e-9 else 1e-9)
    for i in range(m):
        mat[i][i] += penalty

    for col in range(m):
        pivot = max(range(col, m), key=lambda r: abs(mat[r][col]))
        if pivot != col:
            mat[col], mat[pivot] = mat[pivot], mat[col]
        diag = mat[col][col]
        if abs(diag) < 1e-12:
            continue
        for c in range(col, m + 1):
            mat[col][c] /= diag
        for r in range(m):
            if r == col:
                continue
            factor = mat[r][col]
            if factor == 0.0:
                continue
            for c in range(col, m + 1):
                mat[r][c] -= factor * mat[col][c]
    return [mat[i][m] for i in range(m)]


def _fit_model(points, indices, lam: float):
    rows = [points[i]["row"] for i in indices]
    targets_x = [points[i]["x"] for i in indices]
    targets_y = [points[i]["y"] for i in indices]
    mean, std = make_scaler(rows)
    normalized = [apply_scaler(mean, std, row) for row in rows]
    return {
        "mean": mean,
        "std": std,
        "coef_x": solve_ridge(normalized, targets_x, lam),
        "coef_y": solve_ridge(normalized, targets_y, lam),
    }


def _loo_residuals(points, indices, lam: float) -> list[float]:
    residuals = []
    for i, index in enumerate(indices):
        rest = [j for j in indices if j != index]
        if len(rest) < FEATURE_COUNT:
            residuals.append(0.0)
            continue
        model = _fit_model(points, rest, lam)
        row = apply_scaler(model["mean"], model["std"], points[index]["row"])
        px = sum(c * v for c, v in zip(model["coef_x"], row))
        py = sum(c * v for c, v in zip(model["coef_y"], row))
        residuals.append(math.hypot(px - points[index]["x"], py - points[index]["y"]))
    return residuals


def _rms(values) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else 0.0


def calibrate(points, screen_px: float, lam: float = 1e-3) -> dict:
    """キャリブ点から回帰モデルを解く。点は 13 個以上を想定。"""
    indices = list(range(len(points)))
    reference = _rms(_loo_residuals(points, indices, lam))

    # 1点だけ大きく外しているときは、その点を除いた方が良くなるなら捨てる
    dropped = 0
    if len(points) > FEATURE_COUNT:
        best_score, worst = reference, None
        for i, index in enumerate(indices):
            rest = [j for j in indices if j != index]
            score = _rms(_loo_residuals(points, rest, lam))
            if score < best_score:
                best_score, worst = score, index
        if worst is not None and best_score < reference * 0.5:
            indices = [i for i in indices if i != worst]
            dropped = 1

    model = _fit_model(points, indices, lam)
    err_px = _rms(_loo_residuals(points, indices, lam)) * screen_px
    return {**model, "err_px": err_px, "dropped": dropped, "inliers": len(indices)}


class GazeModel:
    """正規化座標 [0,1] を出す推定器。mode で視線と頭の寄与を切り替えられる。"""

    def __init__(self, mean, std, coef_x, coef_y, err_px=0.0, reference=None, eye_geometry=None, meta=None) -> None:
        self.mean = mean
        self.std = std
        self.coef_x = coef_x
        self.coef_y = coef_y
        self.err_px = err_px
        self.reference = reference          # 基準フレームのランドマーク（68点）
        self.eye_geometry = eye_geometry    # build_eye_geometry の結果
        self.meta = meta or {}

    @classmethod
    def from_calibration(cls, fitted, reference, eye_geometry, meta=None) -> "GazeModel":
        return cls(fitted["mean"], fitted["std"], fitted["coef_x"], fitted["coef_y"],
                   fitted["err_px"], reference, eye_geometry, meta)

    def predict(self, row, mode: str = MODE_BOTH):
        use_gaze = mode != MODE_HEAD
        use_head = mode != MODE_GAZE
        x = y = 0.0
        for j in range(FEATURE_COUNT):
            is_eye = j in (F_EYE_H, F_EYE_V, F_EYE_H2, F_EYE_V2)
            is_head = j in (F_YAW, F_PITCH)
            if (is_eye and not use_gaze) or (is_head and not use_head):
                continue
            z = 1.0 if j == 0 else (row[j] - self.mean[j]) / self.std[j]
            # キャリブで見ていない領域まで外挿すると暴れるので ±4σ で頭打ち
            z = min(max(z, -4.0), 4.0)
            x += self.coef_x[j] * z
            y += self.coef_y[j] * z
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        return (clamp01(x), clamp01(y))


# ---------------------------------------------------------------------------
# 制御（仮想カーソル側）
# ---------------------------------------------------------------------------


class CursorController:
    """観測を一定周期で処理して、仮想カーソルの移動と注視クリックを決める。

    実カーソルは一切触らない。出力は「相対移動量」と「クリック」だけ。
    """

    DISARMED = "disarmed"
    ARMED = "armed"
    PAUSED = "paused"
    STOPPED = "stopped"

    def __init__(self, config: dict | None = None) -> None:
        cfg = {
            "dwell_ms": 900.0,
            "dwell_radius_px": 48.0,
            "dwell_rearm_factor": 2.5,
            "dwell_cooldown_ms": 700.0,
            "max_step_px": 220.0,
            "max_target_speed_px_per_sec": 2500.0,
            "deadzone_px": 40.0,
            "pause_after_ms": 400.0,
            "fail_stop_after_ms": 2500.0,
            "latency_samples": 30,
        }
        cfg.update(config or {})
        self.config = cfg

        self.state = self.DISARMED
        self.reason = ""
        self.width_px = 0.0
        self.height_px = 0.0
        self.gaze = (0.5, 0.5)
        self.gaze_at = -1e9
        self.gaze_sample_at = -1e9
        self.has_gaze = False
        self.target = (0.0, 0.0)
        self.target_ready = False
        self.last_tick_at = 0.0
        self.pos = (0.0, 0.0)
        self.dwell_anchor = (0.0, 0.0)
        self.dwell_since = 0.0
        self.dwell_ready = False
        self.dwell_cooldown = False
        self.dwell_last_click = -1e9
        self.dwell_progress = 0.0
        self.latencies: list[float] = []
        self.latency_ms: float | None = None
        self.moves = 0
        self.clicks = 0
        self.moved_px = 0.0
        self.click_requested = False

    # --- 入力 ---

    def arm(self, now_ms: float, width_px: float, height_px: float, position=(0.0, 0.0)) -> None:
        self.width_px = width_px
        self.height_px = height_px
        self.state = self.ARMED
        self.reason = ""
        self.gaze_at = now_ms
        self.has_gaze = False
        self.target_ready = False
        self.last_tick_at = now_ms
        self.pos = tuple(position)
        self.dwell_ready = False
        self.dwell_cooldown = False
        self.dwell_progress = 0.0

    def stop(self, reason: str = "stopped") -> None:
        self.state = self.STOPPED
        self.reason = reason
        self.dwell_progress = 0.0
        self.dwell_cooldown = False

    def observe_gaze(self, now_ms: float, sample_ms: float, x: float, y: float) -> None:
        self.gaze = (clamp01(x), clamp01(y))
        self.gaze_at = now_ms
        self.gaze_sample_at = sample_ms
        self.has_gaze = True
        if self.state == self.PAUSED:
            self.state = self.ARMED
            self.reason = ""

    def set_position(self, position) -> None:
        """仮想カーソルの現在位置（アプリ側が知っている）。"""
        self.pos = tuple(position)

    # --- 制御（一定周期で呼ぶ） ---

    def tick(self, now_ms: float) -> tuple[float, float]:
        if self.state in (self.DISARMED, self.STOPPED):
            return (0.0, 0.0)

        age = now_ms - self.gaze_at
        if age > self.config["fail_stop_after_ms"]:
            self.state = self.STOPPED
            self.reason = "gaze lost"
            self.dwell_progress = 0.0
            return (0.0, 0.0)
        if not self.has_gaze or age > self.config["pause_after_ms"]:
            if self.state == self.ARMED:
                self.state = self.PAUSED
                self.reason = "no gaze" if self.has_gaze else "waiting for gaze"
            self.dwell_progress = 0.0
            self.dwell_ready = False
            return (0.0, 0.0)

        latency = now_ms - self.gaze_sample_at
        if 0.0 <= latency < 1000.0:
            self.latencies.append(latency)
            if len(self.latencies) > self.config["latency_samples"]:
                self.latencies.pop(0)
            self.latency_ms = sorted(self.latencies)[len(self.latencies) // 2]

        # 目標は生の推定値をそのまま使わず、移動速度を制限して追従させる
        raw = (self.gaze[0] * self.width_px, self.gaze[1] * self.height_px)
        if not self.target_ready:
            self.target = raw
            self.target_ready = True
        else:
            dt_sec = min(max((now_ms - self.last_tick_at) / 1000.0, 1e-3), 0.1)
            max_step = self.config["max_target_speed_px_per_sec"] * dt_sec
            dx, dy = raw[0] - self.target[0], raw[1] - self.target[1]
            distance = math.hypot(dx, dy)
            if distance > max_step:
                k = max_step / distance
                self.target = (self.target[0] + dx * k, self.target[1] + dy * k)
            else:
                self.target = raw
        self.last_tick_at = now_ms

        # 不感帯より内側なら動かさない（推定の細かい揺れを追わない）
        error_x, error_y = self.target[0] - self.pos[0], self.target[1] - self.pos[1]
        distance = math.hypot(error_x, error_y)
        step = (0.0, 0.0)
        if distance > self.config["deadzone_px"]:
            scale = min(1.0, self.config["max_step_px"] / distance)
            step = (round(error_x * scale), round(error_y * scale))
            if step[0] or step[1]:
                self.moves += 1
                self.moved_px += math.hypot(*step)

        self._update_dwell(now_ms)
        return step

    def _update_dwell(self, now_ms: float) -> None:
        radius = self.config["dwell_radius_px"]
        rearm = radius * self.config["dwell_rearm_factor"]
        if not self.dwell_ready:
            self.dwell_anchor = self.target
            self.dwell_since = now_ms
            self.dwell_ready = True
            self.dwell_progress = 0.0
            return

        to_anchor = math.hypot(self.target[0] - self.dwell_anchor[0], self.target[1] - self.dwell_anchor[1])
        if self.dwell_cooldown:
            if to_anchor > rearm and now_ms - self.dwell_last_click > self.config["dwell_cooldown_ms"]:
                self.dwell_cooldown = False
                self.dwell_anchor = self.target
                self.dwell_since = now_ms
            self.dwell_progress = 0.0
            return

        if to_anchor > rearm:
            self.dwell_anchor = self.target
            self.dwell_since = now_ms
            self.dwell_progress = 0.0
            return

        elapsed = now_ms - self.dwell_since
        self.dwell_progress = clamp01(elapsed / self.config["dwell_ms"])
        if elapsed >= self.config["dwell_ms"] and to_anchor <= radius:
            self.dwell_cooldown = True
            self.dwell_last_click = now_ms
            self.dwell_progress = 0.0
            self.clicks += 1
            self.click_requested = True
