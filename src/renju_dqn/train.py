"""DQN training loop: TD-error updates against a target network, fed by a replay
buffer that is warmed up from `mcts.cpp` self-play logs and then grown online via
epsilon-greedy self-play (Plan.md phase4).

Self-play and gradient updates run on separate threads (actor/learner): self-play is
CPU-bound (native rule checks) while training is GPU-bound, so overlapping them instead of
alternating in one loop lets both proceed concurrently.
"""

from __future__ import annotations

import queue
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm import tqdm

from .board_encoder import encode_board
from .dataset import ReplayDataset
from .env import SelectMove, play_self_play_game
from .model import RenjuResNetDQN
from .replay_buffer import ReplayBuffer
from .rules import BOARD_CELLS
from .utils import ensure_mlflow_experiment, flatten_config, select_device, set_seed


def build_model(cfg: DictConfig) -> RenjuResNetDQN:
    return RenjuResNetDQN(
        in_channels=cfg.model.in_channels,
        channels=cfg.model.channels,
        num_blocks=cfg.model.num_blocks,
        head_channels=cfg.model.head_channels,
        num_move_labels=cfg.model.num_move_labels,
        dueling=cfg.model.dueling,
        noisy=cfg.train.exploration == "noisy_net",
    )


def build_optimizer(model: nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    if cfg.optimizer.name != "adamw":
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer.name}")
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        betas=tuple(cfg.optimizer.betas),
        eps=cfg.optimizer.eps,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: DictConfig):
    if cfg.scheduler.name == "none":
        return None
    raise ValueError(f"Unsupported scheduler: {cfg.scheduler.name}")


def epsilon_for_curriculum(
    window: int | None, board_cells: int, epsilon_start: float, epsilon_end: float
) -> float:
    """Anneal epsilon by curriculum progress instead of a separately-tuned step count: full
    exploration while the window covers none of the game, decaying to `epsilon_end` exactly as
    the window reaches `board_cells`. Keeps self-play exploring for as long as curriculum training
    hasn't reached the states self-play is generating, and tracks curriculum_step_size /
    curriculum_epoch_interval automatically if they change.
    """
    if window is None:
        return epsilon_end
    fraction_covered = min(1.0, window / board_cells)
    return epsilon_start + fraction_covered * (epsilon_end - epsilon_start)


def curriculum_window_for_epoch(epoch: int, step_size: int, epoch_interval: int) -> int | None:
    """Steps-from-terminal window for `epoch` (1-indexed): opens `step_size` more steps every
    `epoch_interval` epochs (epoch 1..interval -> step_size, interval+1..2*interval -> 2*step_size,
    ...). Returns `None` (no filtering) once `step_size` is non-positive.
    """
    if step_size <= 0:
        return None
    interval = max(1, epoch_interval)
    stage = -(-epoch // interval)  # ceil division
    return step_size * stage


def make_epsilon_greedy_policy(
    model: nn.Module,
    device: torch.device,
    epsilon: float,
    generator: torch.Generator,
    model_lock: threading.Lock | None = None,
) -> SelectMove:
    """Build a `select_move` callback for `env.play_self_play_game` at a fixed epsilon.

    `model_lock`, if given, is held around the forward pass -- needed when `model`'s weights can
    be swapped out from under it by another thread (the learner syncing the target network).
    """

    def select_move(board: list[int], player: int, prev_move: int | None, mask: list[bool]) -> int:
        legal_indices = [index for index, ok in enumerate(mask) if ok]
        if torch.rand((), generator=generator).item() < epsilon:
            choice = int(torch.randint(0, len(legal_indices), (1,), generator=generator).item())
            return legal_indices[choice]

        state = encode_board(board, player, prev_move).unsqueeze(0).to(device)
        legal_mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device)
        model.eval()
        with torch.no_grad():
            if model_lock is not None:
                with model_lock:
                    q_values = model(state).squeeze(0)
            else:
                q_values = model(state).squeeze(0)
        q_values = q_values.masked_fill(~legal_mask_tensor, float("-inf"))
        return int(q_values.argmax().item())

    return select_move


def make_noisy_net_policy(
    model: RenjuResNetDQN,
    device: torch.device,
    model_lock: threading.Lock | None = None,
) -> SelectMove:
    """Build a `select_move` callback that always plays the greedy argmax move, relying on
    `model`'s NoisyLinear head (see model.py) for exploration instead of epsilon-greedy
    randomness. Resamples the noise before every move so each decision sees a fresh sample.
    """

    def select_move(board: list[int], player: int, prev_move: int | None, mask: list[bool]) -> int:
        state = encode_board(board, player, prev_move).unsqueeze(0).to(device)
        legal_mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device)
        # eval() for batchnorm's benefit (single-sample inference), but the noisy head's
        # own noise still needs to be switched on -- that's exploration here, not epsilon.
        model.eval()
        model.set_noise_training(True)
        with torch.no_grad():
            if model_lock is not None:
                with model_lock:
                    model.reset_noise()
                    q_values = model(state).squeeze(0)
            else:
                model.reset_noise()
                q_values = model(state).squeeze(0)
        q_values = q_values.masked_fill(~legal_mask_tensor, float("-inf"))
        return int(q_values.argmax().item())

    return select_move


class _SharedValue:
    """A value shared between threads (epsilon, curriculum window), written once per epoch by
    the learner and read by the actor/prefetch threads.
    """

    def __init__(self, value: Any) -> None:
        self._value = value
        self._lock = threading.Lock()

    def get(self) -> Any:
        with self._lock:
            return self._value

    def set(self, value: Any) -> None:
        with self._lock:
            self._value = value


class _ThreadSafeCounter:
    """Self-play games completed so far, for progress/logging only."""

    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    def increment(self) -> None:
        with self._lock:
            self._value += 1

    @property
    def value(self) -> int:
        with self._lock:
            return self._value


def self_play_worker(
    stop_event: threading.Event,
    epsilon_box: _SharedValue,
    target_model: RenjuResNetDQN,
    model_lock: threading.Lock,
    device: torch.device,
    replay_buffer: ReplayBuffer,
    generator: torch.Generator,
    games_played: _ThreadSafeCounter,
    games_budget_box: _SharedValue,
    exploration: str,
) -> None:
    """Actor loop: self-plays against `target_model` and pushes finished games into
    `replay_buffer` until `stop_event` is set. Runs on its own thread so it can overlap with
    the learner thread's GPU-bound gradient updates instead of alternating with them.

    Uses `target_model` (only ever wholesale-replaced via `load_state_dict`, never
    gradient-updated in place) rather than the online model, so the only thing `model_lock` has
    to guard is that swap -- the learner's optimizer steps on the online model need no
    coordination with this thread at all.

    `games_budget_box` holds `(epoch_start_games, max_games_per_epoch)`, refreshed once per
    epoch by the learner loop. Left uncapped (`max_games_per_epoch is None`), this thread runs
    as fast as it can, so the self-play/gradient-update ratio -- and therefore the buffer's
    state-action distribution -- drifts with however fast this thread happens to run relative to
    the learner. Once the epoch's budget is used up, this thread idles until the next epoch
    raises it.

    `exploration` selects how moves are picked: "epsilon_greedy" mixes in uniformly random
    legal moves per `epsilon_box`; "noisy_net" always plays greedy against `target_model`'s
    NoisyLinear head, which is resampled before every move instead.
    """
    while not stop_event.is_set():
        epoch_start_games, max_games_per_epoch = games_budget_box.get()
        if max_games_per_epoch is not None and games_played.value - epoch_start_games >= max_games_per_epoch:
            stop_event.wait(0.05)
            continue
        if exploration == "noisy_net":
            select_move = make_noisy_net_policy(target_model, device, model_lock=model_lock)
        else:
            select_move = make_epsilon_greedy_policy(
                target_model, device, epsilon_box.get(), generator, model_lock=model_lock
            )
        rows, winner = play_self_play_game(select_move)
        replay_buffer.push_game(rows, winner)
        games_played.increment()


def batch_prefetch_worker(
    stop_event: threading.Event,
    window_box: _SharedValue,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
    out_queue: queue.Queue,
) -> None:
    """Producer thread: keeps `out_queue` filled with ready-to-train, already-on-device batches.

    `replay_buffer.sample()` is the CPU-bound step (native rule checks per transition) that
    otherwise leaves the GPU idle between gradient steps; running it on its own thread(s) lets
    it overlap with the learner's forward/backward pass instead of blocking it.
    """
    while not stop_event.is_set():
        window = window_box.get()
        state, action, reward, next_state, done, next_legal_mask = replay_buffer.sample(
            batch_size, generator=generator, max_steps_from_end=window
        )
        batch = (
            state.to(device),
            action.to(device),
            reward.to(device),
            next_state.to(device),
            done.to(device),
            next_legal_mask.to(device),
        )
        while not stop_event.is_set():
            try:
                out_queue.put(batch, timeout=1.0)
                break
            except queue.Full:
                continue


Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def dqn_update(
    online_model: RenjuResNetDQN,
    target_model: RenjuResNetDQN,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    batch: Batch,
    gamma: float,
    gradient_clip_norm: float | None,
    double_dqn: bool = False,
) -> tuple[float, float]:
    """One TD-error gradient step on a pre-fetched, already-on-device `batch`; returns
    `(loss, mean_q_taken)`.

    `double_dqn` switches the bootstrap target from vanilla DQN -- `target_model` both selects
    and evaluates the next action, which is prone to max-Q overestimation bias -- to Double DQN,
    where `online_model` selects the next action and `target_model` only evaluates it.

    Resamples `online_model`'s NoisyLinear noise once per step (a no-op if it has none), shared
    across its two forward calls below. `target_model` stays in eval() for its whole lifetime
    (see train_model), so its NoisyLinear head -- if any -- always uses its mean weights: a
    deterministic bootstrap target is more stable than one perturbed by the target network's
    own noise.
    """
    state, action, reward, next_state, done, next_legal_mask = batch

    online_model.reset_noise()

    with torch.no_grad():
        next_q_target = target_model(next_state)
        next_q_target = next_q_target.masked_fill(~next_legal_mask, float("-inf"))
        if double_dqn:
            was_training = online_model.training
            online_model.eval()
            next_q_online = online_model(next_state).masked_fill(~next_legal_mask, float("-inf"))
            online_model.train(was_training)
            next_action = next_q_online.argmax(dim=1, keepdim=True)
            next_q_max = next_q_target.gather(1, next_action).squeeze(1)
        else:
            next_q_max = next_q_target.max(dim=1).values
        next_q_max = torch.where(done, torch.zeros_like(next_q_max), next_q_max)
        # next_state is encoded from the opponent's perspective, so their best value is negated.
        target = reward - gamma * next_q_max

    online_model.train()
    q_values = online_model(state)
    q_taken = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

    loss = criterion(q_taken, target)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if gradient_clip_norm is not None and gradient_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(online_model.parameters(), gradient_clip_norm)
    optimizer.step()

    return loss.item(), q_taken.mean().item()


def train_model(cfg: DictConfig) -> None:
    if cfg.train.exploration not in ("epsilon_greedy", "noisy_net"):
        raise ValueError(f"Unsupported exploration strategy: {cfg.train.exploration}")

    set_seed(cfg.seed)
    device = select_device(cfg.train.device)
    # Each thread that samples/self-plays gets its own generator: torch.Generator isn't
    # thread-safe to share.
    actor_generator = torch.Generator().manual_seed(cfg.seed + 1)
    model_lock = threading.Lock()

    online_model = build_model(cfg).to(device)
    target_model = build_model(cfg).to(device)
    target_model.load_state_dict(online_model.state_dict())
    target_model.eval()
    for param in target_model.parameters():
        param.requires_grad_(False)

    optimizer = build_optimizer(online_model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    criterion = nn.SmoothL1Loss()

    replay_buffer = ReplayBuffer(
        capacity=cfg.train.replay_buffer_capacity,
        gamma=cfg.train.gamma,
        coefficient=cfg.train.reward_shaping_coefficient,
    )
    warmup_dataset = ReplayDataset(
        cfg.data.path,
        gamma=cfg.train.gamma,
        coefficient=cfg.train.reward_shaping_coefficient,
        max_rows=cfg.data.max_rows,
    )
    replay_buffer.warmup_from_dataset(warmup_dataset)
    if len(replay_buffer) < cfg.train.batch_size:
        raise ValueError(
            f"Warmup buffer ({len(replay_buffer)} transitions) is smaller than "
            f"batch_size ({cfg.train.batch_size})."
        )

    output_root = Path(cfg.train.output_root)
    checkpoint_dir = output_root / cfg.train.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_dir = output_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    resolved_config_path = config_dir / "resolved_config.yaml"
    OmegaConf.save(cfg, resolved_config_path, resolve=True)

    ensure_mlflow_experiment(
        tracking_uri=cfg.mlflow.tracking_uri,
        experiment_name=cfg.mlflow.experiment_name,
        artifact_root=cfg.mlflow.artifact_root,
    )
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    run_name = f"{cfg.mlflow.run_name_prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    best_epoch_loss = float("inf")
    best_checkpoint_path = checkpoint_dir / cfg.train.checkpoint_name
    best_model_state: dict[str, torch.Tensor] | None = None
    global_step = 0

    # Namespaced by run_name (unlike best_checkpoint_path) so periodic checkpoints from
    # different runs never collide/overwrite each other.
    periodic_checkpoint_dir = checkpoint_dir / run_name
    if cfg.train.checkpoint_every_epochs:
        periodic_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    stop_event = threading.Event()
    games_played = _ThreadSafeCounter()
    initial_window = curriculum_window_for_epoch(
        1, cfg.train.curriculum_step_size, cfg.train.curriculum_epoch_interval
    )
    epsilon_box = _SharedValue(
        epsilon_for_curriculum(
            initial_window, BOARD_CELLS, cfg.train.epsilon_start, cfg.train.epsilon_end
        )
    )
    window_box = _SharedValue(initial_window)
    games_budget_box = _SharedValue((0, cfg.train.max_games_per_epoch))
    actor_thread = threading.Thread(
        target=self_play_worker,
        args=(
            stop_event,
            epsilon_box,
            target_model,
            model_lock,
            device,
            replay_buffer,
            actor_generator,
            games_played,
            games_budget_box,
            cfg.train.exploration,
        ),
        name="self-play-actor",
        daemon=True,
    )

    # Batch prefetching: replay_buffer.sample() is CPU-bound (native rule checks per
    # transition), so several threads each running it in parallel (native calls release the
    # GIL) keep the queue fed fast enough that the learner is never waiting on the GPU's next
    # batch instead of the other way around.
    batch_queue: queue.Queue[Batch] = queue.Queue(maxsize=cfg.train.prefetch_workers * 2)
    prefetch_threads = [
        threading.Thread(
            target=batch_prefetch_worker,
            args=(
                stop_event,
                window_box,
                replay_buffer,
                cfg.train.batch_size,
                device,
                torch.Generator().manual_seed(cfg.seed + 100 + worker_index),
                batch_queue,
            ),
            name=f"batch-prefetch-{worker_index}",
            daemon=True,
        )
        for worker_index in range(cfg.train.prefetch_workers)
    ]

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(flatten_config(cfg))
        mlflow.log_artifact(str(resolved_config_path), artifact_path="configs")

        actor_thread.start()
        for prefetch_thread in prefetch_threads:
            prefetch_thread.start()
        try:
            for epoch in range(1, cfg.train.max_epochs + 1):
                curriculum_window = curriculum_window_for_epoch(
                    epoch, cfg.train.curriculum_step_size, cfg.train.curriculum_epoch_interval
                )
                epsilon = epsilon_for_curriculum(
                    curriculum_window, BOARD_CELLS, cfg.train.epsilon_start, cfg.train.epsilon_end
                )
                epsilon_box.set(epsilon)
                window_box.set(curriculum_window)
                games_budget_box.set((games_played.value, cfg.train.max_games_per_epoch))
                epoch_loss_total = 0.0
                epoch_q_total = 0.0
                epoch_updates = 0
                epoch_progress = tqdm(
                    range(cfg.train.steps_per_epoch),
                    desc=(
                        f"Epoch {epoch}/{cfg.train.max_epochs} "
                        f"[learner{f', last {curriculum_window} steps' if curriculum_window is not None else ''}]"
                    ),
                    leave=True,
                    dynamic_ncols=True,
                    file=sys.stdout,
                )

                for _ in epoch_progress:
                    batch = batch_queue.get()
                    loss, mean_q = dqn_update(
                        online_model,
                        target_model,
                        optimizer,
                        criterion,
                        batch,
                        cfg.train.gamma,
                        cfg.train.gradient_clip_norm,
                        double_dqn=cfg.train.double_dqn,
                    )
                    if scheduler is not None:
                        scheduler.step()

                    global_step += 1
                    epoch_loss_total += loss
                    epoch_q_total += mean_q
                    epoch_updates += 1

                    if global_step % cfg.train.target_update_interval == 0:
                        with model_lock:
                            target_model.load_state_dict(online_model.state_dict())

                    if global_step % cfg.train.log_every_steps == 0:
                        mlflow.log_metric("train_step_td_loss", loss, step=global_step)
                        mlflow.log_metric("train_step_mean_q", mean_q, step=global_step)
                        if cfg.train.exploration == "epsilon_greedy":
                            mlflow.log_metric("epsilon", epsilon, step=global_step)

                    epoch_progress.set_postfix(
                        td_loss=f"{epoch_loss_total / epoch_updates:.4f}",
                        mean_q=f"{epoch_q_total / epoch_updates:.4f}",
                        buffer=len(replay_buffer),
                        games=games_played.value,
                        **(
                            {"epsilon": f"{epsilon:.3f}"}
                            if cfg.train.exploration == "epsilon_greedy"
                            else {"exploration": "noisy_net"}
                        ),
                    )

                epoch_progress.close()
                epoch_metrics = {
                    "td_loss": epoch_loss_total / epoch_updates,
                    "mean_q": epoch_q_total / epoch_updates,
                }
                mlflow.log_metric("train_td_loss", epoch_metrics["td_loss"], step=epoch)
                mlflow.log_metric("train_mean_q", epoch_metrics["mean_q"], step=epoch)
                mlflow.log_metric("replay_buffer_size", len(replay_buffer), step=epoch)
                mlflow.log_metric("self_play_games", games_played.value, step=epoch)

                is_best = epoch_metrics["td_loss"] < best_epoch_loss
                if is_best:
                    best_epoch_loss = epoch_metrics["td_loss"]
                    best_model_state = {
                        key: value.detach().cpu() for key, value in online_model.state_dict().items()
                    }
                    checkpoint = {
                        "model_state_dict": best_model_state,
                        "config": OmegaConf.to_container(cfg, resolve=True),
                        "epoch": epoch,
                        "train_td_loss": best_epoch_loss,
                    }
                    torch.save(checkpoint, best_checkpoint_path)
                    mlflow.log_metric("best_train_td_loss", best_epoch_loss, step=epoch)

                print(
                    f"epoch={epoch} "
                    f"train_td_loss={epoch_metrics['td_loss']:.4f} "
                    f"train_mean_q={epoch_metrics['mean_q']:.4f} "
                    f"best_train_td_loss={best_epoch_loss:.4f} "
                    f"self_play_games={games_played.value}",
                    flush=True,
                )

                if is_best:
                    print(
                        f"best_checkpoint_updated={best_checkpoint_path.resolve()} "
                        f"train_td_loss={best_epoch_loss:.4f}",
                        flush=True,
                    )

                if cfg.train.checkpoint_every_epochs and epoch % cfg.train.checkpoint_every_epochs == 0:
                    periodic_checkpoint_path = periodic_checkpoint_dir / f"epoch_{epoch:04d}.pt"
                    torch.save(
                        {
                            "model_state_dict": {
                                key: value.detach().cpu()
                                for key, value in online_model.state_dict().items()
                            },
                            "config": OmegaConf.to_container(cfg, resolve=True),
                            "epoch": epoch,
                            "train_td_loss": epoch_metrics["td_loss"],
                        },
                        periodic_checkpoint_path,
                    )
                    print(f"periodic_checkpoint_saved={periodic_checkpoint_path.resolve()}", flush=True)
        finally:
            stop_event.set()
            actor_thread.join(timeout=30)
            for prefetch_thread in prefetch_threads:
                prefetch_thread.join(timeout=30)

        if best_model_state is None:
            raise RuntimeError("Training completed without producing a checkpoint.")

        mlflow.log_artifact(str(best_checkpoint_path), artifact_path="checkpoints")

        if cfg.mlflow.log_model:
            best_model = build_model(cfg)
            best_model.load_state_dict(best_model_state)
            best_model.eval()
            mlflow.pytorch.log_model(best_model, name="model")

    print(f"best_checkpoint={best_checkpoint_path.resolve()}", flush=True)
