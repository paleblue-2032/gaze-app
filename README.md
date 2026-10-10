# gaze-app

Webカメラの映像から**目線と頭の向き**を推定して、**アプリ内の仮想カーソル**を動かすデスクトップアプリ。

- 画面はアプリのウィンドウの中だけで完結する。**実カーソルは一切操作しない**（ジャックしない）
- 顔の平行移動・距離変化・傾きを補正するので、少々頭が動いてもズレにくい
- Python + Tkinter。依存は Nix flake で管理（Nix が無い環境向けの手順も下にある）

## できること

- 目線と頭の向きで仮想カーソルを動かし、**900ms 注視するとクリック**
- 13点のキャリブレーション。最初の中央で「正面を向いていた姿勢」を基準として保存し、
  毎フレーム最小二乗で相似変換（平行移動・拡大縮小・回転）を解いてから眼球特徴を測る
  → **頭が動いても眼球特徴が崩れない**
- 頭の向きの代理量（鼻の横ずれ＝yaw、顎と眉の縦距離＝pitch）も特徴に入れた統合推定。
  `MODE` で「視線+頭 / 視線のみ / 頭のみ」を切り替えて効きを比較できる
- 瞬き・瞳孔を見失ったフレーム・急に飛んだ検出は捨てて前の値を保持。
  制御は 100Hz 固定で **40px の不感帯**を持ち、推定の細かい揺れを追わない

## 動作環境

- Linux（開発・動作確認は NixOS + niri / XWayland）
- カメラ（`/dev/videoN`）
- Nix（flakes 有効）または Python 3.11+ と dlib / OpenCV / numpy / Tk
- dlib の学習済みモデル `shape_predictor_68_face_landmarks.dat`（初回に取得。約100MB）

## 導入と実行

### Nix を使う場合（推奨）

```sh
git clone <this repo> gaze-app
cd gaze-app
nix run . --windowed      # まずウィンドウで動作確認
nix run .                 # 全画面
```

開発するときは `nix develop` してから `python3 gaze_app.py`。

> flake は git の**追跡ファイルだけ**をコピーする。新しいファイルを足したら `git add` してから
> `nix run .` を実行すること（追跡していないファイルは見えない）。

モデルは初回に次のどちらかで取得する（`~/.cache/gaze-app/` に置かれる）:

```sh
nix run . -- --fetch-model
# 手動でやる場合
mkdir -p ~/.cache/gaze-app
curl -sSL https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2 \
  | bzip2 -d > ~/.cache/gaze-app/shape_predictor_68_face_landmarks.dat
```

### Nix を使わない場合

```sh
pip install numpy opencv-python dlib        # Tk は環境によって python3-tk など
python3 gaze_app.py --windowed
```

`gaze-app.desktop` も同梱してある。`~/.local/share/applications/` に置くか、
home-manager / NixOS の設定から `nix run` を呼ぶようにするとランチャから起動できる。

## 操作

画面下のボタンは**実マウスで普通に押せる**（カーソルを奪わないので）。キーでも動く。

| 操作 | 内容 |
| --- | --- |
| `START` | カメラを起動 |
| `CALIBRATE` / `c` | 13点のキャリブレーション。最初の中央は**正面の基準**を取るので頭をまっすぐに |
| `TRACKING` / `t` | 追跡の入／切 |
| `MODE` / `m` | 視線+頭 / 視線のみ / 頭のみ の切替 |
| `DEBUG` / `d` | カメラを画面全体に薄く重ね、検出した目と瞳孔を表示 |
| `RESET` | 的のカウントを戻す |
| `EXIT` / `Esc` | 終了 |

オプション: `--camera N`（カメラ選択）/ `--windowed` / `--self-test`（カメラと検出の確認だけ）/
`--smoke-test`（GUI の組み立てだけ、カメラは開かない）/ `--fetch-model`。

上手く動かないときは:

1. `nix run . -- --self-test` で「顔を検出できたか／虹彩まで取れているか／fps」を確認
2. アプリで `d` を押し、画面左下の表示を見る
   - `eye1 検出できず` → 明るさ・距離・カメラ選択（`--camera 1`）を見直す
   - `eye1 ○ 倍率5.3 面積97 …` → 検出はできている。キャリブレーションを見直す
3. HUD の `RANGE` は「キャリブレーション時に見ていた範囲の外を見ている割合」。
   高いままなら、キャリブ時と同じ位置・姿勢で座り直す

## しくみ

```
gaze_vision.py  カメラ → dlib 68点 → 目を切り出して拡大し瞳孔/虹彩の重心を検出 → 「観測」
       ↓
gaze_core.py    観測 → 基準フレームへの相似変換 → 眼球特徴 → 回帰 → 視線（0..1）
       ↓
gaze_app.py     視線 → 制御（不感帯・目標の速度制限・注視クリック・フェイルストップ）→ 仮想カーソル
```

- 瞳孔検出は「目を切り出して拡大 → 黒帽で暗い塊を強調 → 輪郭を円らしさ・大きさ・中央らしさで採点」。
  実画像では目が 30〜40px しかなく、そのまま閾値処理すると睫毛や陰影と同じスケールになって破綻するため、
  拡大してから処理している。前フレームの位置を手がかりにして別の塊へ乗り換えるのも防いでいる
- キャリブレーションは 13点 × 40フレーム。各点で中央値から外れたフレームを捨ててから平均し、
  リッジ回帰を解く。外れ値の点は leave-one-out で 1 点まで除外する
- 観測が 400ms 途切れたら一時停止、2.5s 途切れたら要再アーム（フェイルストップ）

## テスト

```sh
nix develop --command python3 test_gaze_core.py     # 数値（合成データ）
nix develop --command python3 test_gaze_vision.py   # 顔・瞳孔検出と姿勢補正（実画像・カメラは開かない）
nix develop --command python3 gaze_app.py --smoke-test
```

`test_gaze_vision.py` は `~/.cache/gaze-app/face_test.jpg` を使う（無ければ取得方法を表示する）。
実画像に拡大・回転・平行移動を加えても眼球特徴が保たれること（＝姿勢補正が効いていること）を確認している。

## 制限と今後の課題

- **精度の天井はカメラに映る目の大きさ**で決まる。目が 40px 程度だと瞳孔位置は ±2〜3px 揺れ、
  比率にすると ±0.1 になる。1080p にする・少し近づく・IR 照明を足す、で改善する
- 顔の向きは 2D の代理量なので、大きく横を向くと弱い。3D の剛体合わせ（68点を3次元に拡張）を
  入れる余地がある
- **学習済みモデルのライセンス**: 上の手順で取得する `shape_predictor_68_face_landmarks.dat` は
  iBUG 300-W データセットで学習されており、そのデータセットのライセンスが商用利用を除外しているため、
  モデル自体も商用利用できない（詳細は
  [dlib-models の README](https://github.com/davisking/dlib-models) を参照）。
  モデルファイルは同梱していない（各自で取得する）ので、このリポジトリのコードはこの制約を受けない
- 実カーソルは操作しない。動かすのはこのアプリのウィンドウの中の仮想カーソルだけ

## ファイル

| ファイル | 役割 |
| --- | --- |
| `gaze_core.py` | 推定と制御のコア。numpy も OpenCV も使わない純 Python で、移植しやすい形にしてある |
| `gaze_vision.py` | カメラ・dlib・瞳孔検出（重い依存はここだけ） |
| `gaze_app.py` | Tkinter の画面と状態遷移 |
| `test_gaze_core.py` / `test_gaze_vision.py` | テスト |
| `flake.nix` | 依存の管理。C / Rust のコアを足すときの場所もコメントで用意してある |

キャリブレーション結果と学習済みモデルは `~/.cache/gaze-app/` に置く（`GAZE_APP_HOME` で変更可）。
