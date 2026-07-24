# RenjuDQN

PyTorch + Hydra + MLflow + uv で構成した、五目並べの ResNet DQN（強化学習）プロジェクトです。

> 元リポジトリ: https://github.com/shuichi/RenjuTransformer.git

## セットアップ

```bash
uv sync
```

## 学習

`data.csv`（`config/data/default.yaml` の `path`、既定は `data.csv`）は `mcts.cpp` が出力した 1 行 231 列
（`game_id, ply, player, board(225), move_id, winner, foul_loss`）の CSV を想定します。この CSV でリプレイバッファを
ウォームアップした後、ResNet DQN のオンライン自己対局・学習ループ（[train.py](src/renju_dqn/train.py)）を回します。

Hydra エントリポイントは `renju-dqn.py` で、`mode` の既定値は `train` です。

```bash
uv run python ./renju-dqn.py
```

設定は `config/train/default.yaml` 等に対応し、Hydra のオーバーライド構文でその場で上書きできます。

```bash
uv run python ./renju-dqn.py data.path=data.csv train.max_epochs=100
```

## 合成データ生成

`mcts.cpp` は、Renju のルールと禁じ手を考慮した自己対戦データ生成器です。1 対局分を内部でバッファし、対局終了時に確定した `winner`/`foul_loss` を各手の行にまとめて付与してから、`game_id, ply, player, board(225), move_id, winner, foul_loss` の 1 行 CSV（231 列）を標準出力に書き出します。試合進捗と勝敗結果は標準エラー出力に書き出します。

- `game_id`: 対局ごとの通し番号
- `ply`: その対局内の手数（1 始まり）
- `player`: 着手側（`1`=黒 / `2`=白）
- `board(225)`: 着手前の盤面（`0`=空 / `1`=黒 / `2`=白）
- `move_id`: 着手位置（`0`〜`224`）
- `winner`: 対局の勝者（`0`=引分 / `1`=黒 / `2`=白）
- `foul_loss`: 黒の反則（長連など）による白勝ちなら `1`、それ以外は `0`

`next_state`（次の局面）や `done`（対局の最終手かどうか）は列として持たず、`game_id`/`ply` を使って Python 側（データセット構築ロジック）で再構成・判定します。reward の計算も Python 側に寄せており、C++ 側は `winner`/`foul_loss` のみを出力します。

### ビルド

```bash
g++ -std=c++17 -O2 -pthread ./mcts.cpp -o ./mcts
```

### Usage

数千試合規模でも数時間かかる（例: 10コア/16スレッド機で `--simulations 1000 --parallel 12` なら 3000 試合で9時間程度）ので、`nohup` でバックグラウンド実行してターミナルを閉じても継続させるのが基本です。

```bash
nohup ./mcts 3000 --simulations 1000 --parallel 12 --seed 42 > data.csv 2> error.log &
echo $! > mcts.pid
```

この例では次を行います。

- `2000` 試合の自己対戦を実行
- 1 手あたり `1000` 回の MCTS シミュレーションを実行
- `12` スレッドで試合単位に並列化（コア数・スレッド数に応じて調整。物理コア数を超えても `--parallel 16` あたりまではスループットが伸びる一方、PC を他用途にも使うなら余力を残すため `12` 程度が無難）
- `nohup` によりログアウト・ターミナル終了後も継続。`stdout`/`stderr` を明示的にリダイレクトしているので `nohup.out` は作られない
- 学習用 CSV を `data.csv` に保存（既存ファイルがあれば上書きされる点に注意）
- 進捗と勝敗ログを `error.log` に保存
- プロセス ID を `mcts.pid` に保存（後述の確認・停止に使う）

実行中の確認・停止は次のように行えます。

```bash
tail -f error.log                 # 進捗を追う
ps -p $(cat mcts.pid)             # 実行中か確認（プロセスが見当たらなければ完了）
kill $(cat mcts.pid)              # 途中で止めたい場合
```

主な引数は次です。

- `<games>`: 総試合数
- `--simulations <N>`: 1 手あたりの MCTS シミュレーション回数
- `--parallel <N>`: 並列スレッド数
- `--seed <N>`: 乱数 seed
- `--candidate-limit <N>`: 探索対象に残す候補手の上限
- `--rollout-limit <N>`: rollout の最大手数
- `--exploration <C>`: UCT の探索定数
- `--trace-plies`: 標準エラー出力に各手の進捗も出す

ヘルプは次で表示できます。

```bash
./mcts --help
```

## 推論

`renju_dqn.predict` が実装され次第、学習済み checkpoint と盤面から Q 値上位の着手を出力できるようになります。
現時点では `uv run python ./renju-dqn.py mode=predict ...` は `NotImplementedError` です。

## MLflow

追跡 DB は SQLite、artifact はローカルディレクトリです。

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## 設定

設定はすべて `config/` 配下の Hydra 管理です。

- `config/data/`: データセット
- `config/model/`: ResNet DQN のモデル構成
- `config/train/`: 学習条件
- `config/optimizer/`: 最適化
- `config/scheduler/`: スケジューラ
- `config/mlflow/`: 実験管理
- `config/predict/`: 推論
