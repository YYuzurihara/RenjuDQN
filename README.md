# RenjuTransformer

PyTorch + Hydra + MLflow + uv で構成した、五目並べの「次の1手」予測用 TransformerEncoder プロジェクトです。

> 元リポジトリ: https://github.com/shuichi/RenjuTransformer.git

## セットアップ

```powershell
uv sync
```

## 学習

`data.csv` は `mcts.cpp` が出力した 1 行 227 列の CSV を想定します。

```powershell
uv run python .\renju-transformer.py
```

設定は Hydra で上書きできます。

```powershell
uv run python .\renju-transformer.py data.path=sample-log.csv train.max_epochs=5 train.batch_size=32 model.d_model=256
```

## 合成データ生成

`mcts.cpp` は、Renju のルールと禁じ手を考慮した自己対戦データ生成器です。各手について、`board(225) + SEP(228) + move_id` の 1 行 CSV を標準出力に書き出します。試合進捗と勝敗結果は標準エラー出力に書き出します。

### ビルド

```powershell
g++ -std=c++17 -O2 -pthread .\mcts.cpp -o .\mcts.exe
```

### Usage

```powershell
.\mcts.exe 100000 --simulations 1000 --parallel 28 > data.csv 2> error.log
```

この例では次を行います。

- `100000` 試合の自己対戦を実行
- 1 手あたり `1000` 回の MCTS シミュレーションを実行
- `28` スレッドで試合単位に並列化
- 学習用 CSV を `output.csv` に保存
- 進捗と勝敗ログを `error.log` に保存

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

```powershell
.\mcts.exe --help
```

## 推論

学習済み checkpoint と盤面を与えると、次の一手 ID を出力します。

```powershell
uv run python .\renju-transformer.py mode=predict predict.checkpoint_path=artifacts/checkpoints/best_model.pt predict.board_path=board.txt
```

`board.txt` は 225 個の `0,1,2` をカンマ区切りで並べた 1 行ファイルです。

## MLflow

追跡 DB は SQLite、artifact はローカルディレクトリです。

```powershell
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## 設定

設定はすべて `config/` 配下の Hydra 管理です。

- `config/data/`: データセット
- `config/model/`: TransformerEncoder
- `config/train/`: 学習条件
- `config/optimizer/`: 最適化
- `config/scheduler/`: スケジューラ
- `config/mlflow/`: 実験管理
- `config/predict/`: 推論
