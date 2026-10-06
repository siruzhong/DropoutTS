"""Sweep batch size while holding each run's best DropoutTS settings fixed.

The script discovers the best completed DropoutTS run for every requested
model/dataset/horizon, copies its DropoutTS callback parameters, and changes
only the training/evaluation batch size. Completed configurations are skipped
by the normal BasicTS checkpoint hash check.

Example:
    conda run -n BasicTS python scripts/run_batch_size_sweep.py \
        --task forecasting --preset representative --batch-sizes 8 16 32 64 128 \
        --gpus 3 5 6 7
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_baselines as forecasting_entry


TASK_METRICS = {
    "forecasting": ("MSE", False),
}

REPRESENTATIVE = {
    "forecasting": {
        ("PatchTST", "ETTh1"),
        ("PatchTST", "Illness"),
        ("iTransformer", "Weather"),
        ("TimeMixer", "ETTh1"),
        ("WPMixer", "SyntheticTS_noise0.5"),
    },
}

CALLBACK_DEFAULTS = {
    "p_min": 0.05,
    "p_max": 0.5,
    "init_alpha": 10.0,
    "init_sensitivity": 1.0,
    "sparsity_weight": 0.0,
    "use_gate": False,
}


@dataclass(frozen=True)
class SourceRun:
    task: str
    model: str
    dataset: str
    input_len: int | None
    output_len: int | None
    num_features: int | None
    metric: float
    batch_size: int
    callback_params: dict
    cfg_path: Path

    @property
    def key(self) -> tuple:
        return self.model, self.dataset, self.input_len, self.output_len


def load_json(path: Path) -> dict | None:
    try:
        with path.open(encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


def short_model_name(name: str) -> str:
    suffix = "ForForecasting"
    return name[: -len(suffix)] if name.endswith(suffix) else name


def detect_task(cfg: dict) -> str | None:
    name = cfg.get("taskflow", {}).get("name", "")
    if "Forecasting" in name:
        return "forecasting"
    return None


def dropout_params(cfg: dict) -> dict | None:
    for callback in cfg.get("callbacks", []):
        if callback.get("name") != "DropoutTSCallback":
            continue
        result = dict(CALLBACK_DEFAULTS)
        for key, value in callback.get("params", {}).items():
            if key not in result:
                continue
            if key == "use_gate":
                result[key] = str(value).lower() == "true"
            else:
                result[key] = float(value)
        return result
    return None


def metric_value(metrics: dict, name: str) -> float | None:
    try:
        value = float(metrics.get("overall", {}).get(name))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def iter_source_runs(checkpoint_root: Path, task: str) -> Iterable[SourceRun]:
    metric_name, _ = TASK_METRICS[task]
    for metrics_path in checkpoint_root.rglob("test_metrics.json"):
        cfg_path = metrics_path.with_name("cfg.json")
        cfg = load_json(cfg_path)
        metrics = load_json(metrics_path)
        if cfg is None or metrics is None or detect_task(cfg) != task:
            continue
        params = dropout_params(cfg)
        value = metric_value(metrics, metric_name)
        if params is None or value is None:
            continue
        model_cfg = cfg.get("model_config", {})
        input_len = model_cfg.get("input_len", cfg.get("input_len"))
        output_len = model_cfg.get("output_len", cfg.get("output_len"))
        yield SourceRun(
            task=task,
            model=short_model_name(cfg.get("model", {}).get("name", "")),
            dataset=cfg.get("dataset_name", ""),
            input_len=int(input_len) if input_len is not None else None,
            output_len=int(output_len) if output_len is not None else None,
            num_features=model_cfg.get("num_features"),
            metric=value,
            batch_size=int(cfg.get("train_batch_size", 0)),
            callback_params=params,
            cfg_path=cfg_path,
        )


def collect_best(runs: Iterable[SourceRun], task: str) -> list[SourceRun]:
    _, higher_is_better = TASK_METRICS[task]
    best: dict[tuple, SourceRun] = {}
    for run in runs:
        current = best.get(run.key)
        if current is None:
            best[run.key] = run
        elif higher_is_better and run.metric > current.metric:
            best[run.key] = run
        elif not higher_is_better and run.metric < current.metric:
            best[run.key] = run
    return sorted(best.values(), key=lambda run: run.key)


def matches(run: SourceRun, args: argparse.Namespace) -> bool:
    if args.preset == "representative" and (run.model, run.dataset) not in REPRESENTATIVE[run.task]:
        return False
    if args.models and run.model not in args.models:
        return False
    if args.datasets and run.dataset not in args.datasets:
        return False
    if args.horizons and run.output_len not in args.horizons:
        return False
    return True


def launch_one(run: SourceRun, batch_size: int, gpu_id: int) -> None:
    kwargs = dict(run.callback_params)
    kwargs.update(enable_dropout_ts=True, batch_size=batch_size)
    forecasting_entry.run_experiment(
        run.model, run.dataset, int(run.num_features), int(run.input_len),
        int(run.output_len), str(gpu_id), train_sample_ratio=1.0, **kwargs
    )


def worker(queue: Queue, run: SourceRun, batch_size: int) -> None:
    gpu_id = queue.get()
    label = f"{run.task}:{run.model}/{run.dataset}"
    if run.output_len is not None:
        label += f"/{run.output_len}"
    try:
        print(f"[Start] {label} batch={batch_size} GPU={gpu_id}", flush=True)
        launch_one(run, batch_size, gpu_id)
        print(f"[Done] {label} batch={batch_size} GPU={gpu_id}", flush=True)
    except Exception as exc:
        print(f"[Error] {label} batch={batch_size}: {exc}", flush=True)
        import traceback
        traceback.print_exc()
        raise
    finally:
        queue.put(gpu_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASK_METRICS), required=True)
    parser.add_argument("--preset", choices=["representative", "all"], default="representative")
    parser.add_argument("--batch-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--horizons", type=int, nargs="+")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    invalid = [size for size in args.batch_sizes if size < 1]
    if invalid:
        raise ValueError(f"Batch sizes must be positive: {invalid}")

    sources = [
        run for run in collect_best(iter_source_runs(args.checkpoint_root, args.task), args.task)
        if matches(run, args)
    ]
    jobs = [
        (run, batch_size)
        for run in sources
        for batch_size in sorted(set(args.batch_sizes))
        if batch_size != run.batch_size
    ]
    print(f"Selected {len(sources)} source runs and {len(jobs)} sweep jobs.")
    for run in sources:
        horizon = (
            f" horizon={run.output_len}"
            if run.task == "forecasting" and run.output_len is not None
            else ""
        )
        print(
            f"  {run.model}/{run.dataset}{horizon}: best={run.metric:.6g}, "
            f"source_batch={run.batch_size}, cfg={run.cfg_path}"
        )
    if args.dry_run:
        return

    gpu_queue: Queue = Queue()
    for gpu_id in args.gpus:
        gpu_queue.put(gpu_id)
    processes: list[Process] = []
    for run, batch_size in jobs:
        process = Process(target=worker, args=(gpu_queue, run, batch_size))
        process.start()
        processes.append(process)
        time.sleep(0.05)
    for process in processes:
        process.join()
    failures = sum(process.exitcode not in (0, None) for process in processes)
    print(f"Finished {len(processes)} jobs; worker process failures={failures}.")


if __name__ == "__main__":
    main()
