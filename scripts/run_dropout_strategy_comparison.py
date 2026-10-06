"""Run the representative dropout-strategy comparison for DropoutTS.

The experiment compares four training strategies at the same dropout locations:

1. Original model defaults.
2. Fixed dropout, selected on validation MSE from p = 0.00, 0.05, ..., 0.50.
3. A learnable global dropout rate shared by every sample.
4. DropoutTS with the paper's default configuration.

The experiment matrix uses TimeFilter, PatchTST, TimeMixer, and Informer on
ETTh1, ILI, and ExchangeRate.

Fixed-rate candidates are trained without test evaluation. Their validation MSE
is averaged across the requested seeds, one p* is selected per
model/dataset/horizon task, and only that checkpoint is evaluated on the test
split. This keeps test results out of hyperparameter selection.

Examples:
    python scripts/run_dropout_strategy_comparison.py --dry-run
    python scripts/run_dropout_strategy_comparison.py
    python scripts/run_dropout_strategy_comparison.py --all-horizons --seeds 42,43,44
    python scripts/run_dropout_strategy_comparison.py --summary-only
"""

import argparse
import glob
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from multiprocessing import Process, Queue

import torch
from torch import nn


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.models.Informer import Informer, InformerConfig
from basicts.models.PatchTST import PatchTSTConfig, PatchTSTForForecasting
from basicts.models.TimeFilter import TimeFilterConfig, TimeFilterForForecasting
from basicts.models.TimeMixer import TimeMixerConfig, TimeMixerForForecasting
from basicts.modules import DropoutTSContext
from basicts.runners.callback import BasicTSCallback, DropoutTSCallback, EarlyStopping
from basicts.runners.callback.dynamic_dropout import replace_dropout_layers


EXPERIMENT_GROUP = "dropout_strategy_comparison"
CHECKPOINT_ROOT = os.path.join(PROJECT_ROOT, "checkpoints", EXPERIMENT_GROUP)

PRIMARY_MODELS = ("TimeFilter", "PatchTST", "TimeMixer", "Informer")
DATASETS = (("ETTh1", 7), ("Illness", 7), ("ExchangeRate", 8))

DATASET_CONFIGS = {
    "ETTh1": {"input_len": 96, "output_lens": (96, 192, 336, 720)},
    "Illness": {"input_len": 24, "output_lens": (24, 36, 48, 60)},
    "ExchangeRate": {"input_len": 96, "output_lens": (96, 192, 336, 720)},
}
REPRESENTATIVE_HORIZONS = {"ETTh1": 96, "Illness": 48, "ExchangeRate": 96}

FIXED_RATES = tuple(round(index * 0.05, 2) for index in range(11))
NUM_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
DATA_NUM_WORKERS = 4

DROPOUT_TS_CONFIG = {
    "p_min": 0.05,
    "p_max": 0.5,
    "init_alpha": 10.0,
    "init_sensitivity": 1.0,
    "sparsity_weight": 0.0,
    "use_gate": False,
}
GLOBAL_DROPOUT_CONFIG = {
    "p_min": 0.05,
    "p_max": 0.5,
    "init_p": 0.1,
}


def unwrap_model(model):
    """Return the underlying model when a parallel wrapper is present."""
    return model.module if hasattr(model, "module") else model


class FixedDropoutCallback(BasicTSCallback):
    """Set every standard dropout layer to one fixed probability."""

    def __init__(self, p):
        if not 0.0 <= p < 1.0:
            raise ValueError(f"p must be in [0, 1), got {p}")
        self.p = float(p)

    def on_train_start(self, runner, **kwargs):
        model = unwrap_model(runner.model)
        count = 0
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = self.p
                count += 1
        if count == 0:
            raise RuntimeError(
                f"{model.__class__.__name__} has no nn.Dropout layers to control."
            )
        runner.logger.info(
            f"[FixedDropout] Set {count} dropout layer(s) to p={self.p:.2f}."
        )


class BoundedGlobalDropout(nn.Module):
    """One learnable dropout probability shared by all samples and layers."""

    def __init__(self, p_min=0.05, p_max=0.5, init_p=0.1):
        super().__init__()
        if not 0.0 <= p_min < p_max < 1.0:
            raise ValueError("Expected 0 <= p_min < p_max < 1.")
        if not p_min <= init_p <= p_max:
            raise ValueError("init_p must lie in [p_min, p_max].")

        normalized = (init_p - p_min) / (p_max - p_min)
        normalized = min(max(normalized, 1e-6), 1.0 - 1e-6)
        init_logit = math.log(normalized / (1.0 - normalized))

        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))

    def rate(self):
        return self.p_min + (self.p_max - self.p_min) * torch.sigmoid(self.logit)


class LearnableGlobalDropoutCallback(BasicTSCallback):
    """Replace dropout layers and drive them with one learned global rate."""

    def __init__(self, p_min=0.05, p_max=0.5, init_p=0.1):
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.init_p = float(init_p)
        self.controller = None
        self.original_forward = None
        self.best_rate = None
        self.best_validation = None

    def on_train_start(self, runner, **kwargs):
        model = unwrap_model(runner.model)
        device = next(model.parameters()).device

        self.controller = BoundedGlobalDropout(
            p_min=self.p_min,
            p_max=self.p_max,
            init_p=self.init_p,
        ).to(device)
        model.add_module("_learnable_global_dropout", self.controller)

        replaced = replace_dropout_layers(model)
        if not replaced:
            raise RuntimeError(
                f"{model.__class__.__name__} has no nn.Dropout layers to control."
            )

        optimizer_ids = {
            id(parameter)
            for group in runner.optimizer.param_groups
            for parameter in group["params"]
        }
        controller_params = [
            parameter
            for parameter in self.controller.parameters()
            if id(parameter) not in optimizer_ids
        ]
        if controller_params:
            runner.optimizer.add_param_group(
                {
                    "params": controller_params,
                    "lr": runner.optimizer.param_groups[0]["lr"],
                }
            )

        self.original_forward = model.forward

        def wrapped_forward(*args, **forward_kwargs):
            if model.training:
                DropoutTSContext.set_rates(self.controller.rate())
            try:
                return self.original_forward(*args, **forward_kwargs)
            finally:
                DropoutTSContext.clear()

        model.forward = wrapped_forward
        runner.logger.info(
            "[LearnableGlobalDropout] Replaced "
            f"{len(replaced)} layer(s); initial p={self.current_rate():.4f}, "
            f"range=[{self.p_min:.2f}, {self.p_max:.2f}]."
        )

    def on_validate_end(self, runner, **kwargs):
        value = runner.meter_pool.get_value(f"val/{runner.target_metric}")
        better = self.best_validation is None
        if not better and runner.metrics_best == "min":
            better = value < self.best_validation
        elif not better:
            better = value > self.best_validation
        if better:
            self.best_validation = float(value)
            self.best_rate = self.current_rate()

    def current_rate(self):
        if self.controller is None:
            return None
        return float(self.controller.rate().detach().cpu())

    def selected_rate(self):
        return self.best_rate if self.best_rate is not None else self.current_rate()


class ValidationMetricRecorder(BasicTSCallback):
    """Persist the best validation metric and experiment metadata."""

    def __init__(self, metadata):
        self.metadata = metadata
        # Assigned after construction so runtime callback state is not serialized
        # into the BasicTS configuration hash.
        self.rate_callback = None

    def on_train_end(self, runner, **kwargs):
        metric_name = f"val/{runner.target_metric}"
        metric_value = runner.best_metrics.get(metric_name)
        if metric_value is None:
            return

        record = {
            **self.metadata,
            "validation_metric": metric_name,
            "validation_mode": runner.metrics_best,
            "validation_value": float(metric_value),
        }
        if self.rate_callback is not None:
            record["learned_global_p"] = self.rate_callback.selected_rate()

        path = os.path.join(runner.ckpt_save_dir, "validation_metrics.json")
        with open(path, "w", encoding="utf-8") as file:
            json.dump(record, file, indent=2)


def get_timestamp_sizes(dataset_name):
    if dataset_name in {"ETTh1", "ETTh2"}:
        return [24, 7, 31, 366]
    if dataset_name in {"ETTm1", "ETTm2"}:
        return [96, 7, 31, 366]
    return [60, 7, 31, 366]


def get_model_config(model_name, input_len, output_len, num_features, dataset_name):
    if model_name == "Informer":
        config = InformerConfig(
            input_len=input_len,
            output_len=output_len,
            label_len=output_len // 2,
            num_features=num_features,
            use_timestamps=True,
            timestamp_sizes=get_timestamp_sizes(dataset_name),
        )
        return Informer, config, True
    if model_name == "PatchTST":
        config = PatchTSTConfig(
            input_len=input_len,
            output_len=output_len,
            num_features=num_features,
        )
        return PatchTSTForForecasting, config, False
    if model_name == "TimeMixer":
        config = TimeMixerConfig(
            input_len=input_len,
            output_len=output_len,
            num_features=num_features,
        )
        return TimeMixerForForecasting, config, False
    if model_name == "TimeFilter":
        config = TimeFilterConfig(
            input_len=input_len,
            output_len=output_len,
            num_features=num_features,
        )
        return TimeFilterForForecasting, config, False
    raise ValueError(f"Unknown model: {model_name}")


def dataset_config(dataset_name):
    return DATASET_CONFIGS[dataset_name]


def output_lens_for(dataset_name, all_horizons):
    if all_horizons:
        return DATASET_CONFIGS[dataset_name]["output_lens"]
    return (REPRESENTATIVE_HORIZONS[dataset_name],)


def build_tasks(all_horizons=True):
    tasks = []
    for model_name in PRIMARY_MODELS:
        for dataset_name, num_features in DATASETS:
            config = dataset_config(dataset_name)
            for output_len in output_lens_for(dataset_name, all_horizons):
                tasks.append(
                    (
                        model_name,
                        dataset_name,
                        num_features,
                        config["input_len"],
                        output_len,
                    )
                )

    return tasks


def rate_name(rate):
    return f"p_{rate:.2f}".replace(".", "p")


def build_experiments(seeds):
    experiments = []
    for seed in seeds:
        experiments.append(
            {
                "variant": "original",
                "config_name": "model_default",
                "seed": seed,
            }
        )
        for rate in FIXED_RATES:
            experiments.append(
                {
                    "variant": "fixed",
                    "config_name": rate_name(rate),
                    "seed": seed,
                    "fixed_p": rate,
                }
            )
        experiments.append(
            {
                "variant": "learnable_global",
                "config_name": "bounded_0p05_0p50_init_0p10",
                "seed": seed,
                **GLOBAL_DROPOUT_CONFIG,
            }
        )
        experiments.append(
            {
                "variant": "dropoutts",
                "config_name": "paper_default",
                "seed": seed,
                **DROPOUT_TS_CONFIG,
            }
        )
    return experiments


def get_batch_size(model_name, dataset_name, output_len):
    if output_len >= 720:
        return 16
    return DEFAULT_BATCH_SIZE


def get_ckpt_save_dir(
    model_class,
    dataset_name,
    input_len,
    output_len,
    variant,
    config_name,
    seed,
):
    return os.path.join(
        CHECKPOINT_ROOT,
        model_class.__name__,
        f"{dataset_name}_L{input_len}_H{output_len}",
        variant,
        config_name,
        f"seed_{seed}",
    )


def make_callbacks(variant, metadata, experiment):
    strategy_callback = None
    if variant == "fixed":
        strategy_callback = FixedDropoutCallback(experiment["fixed_p"])
    elif variant == "learnable_global":
        strategy_callback = LearnableGlobalDropoutCallback(
            p_min=experiment["p_min"],
            p_max=experiment["p_max"],
            init_p=experiment["init_p"],
        )
    elif variant == "dropoutts":
        strategy_callback = DropoutTSCallback(
            p_min=experiment["p_min"],
            p_max=experiment["p_max"],
            init_alpha=experiment["init_alpha"],
            init_sensitivity=experiment["init_sensitivity"],
            sparsity_weight=experiment["sparsity_weight"],
            use_gate=experiment["use_gate"],
            enable_visualization=False,
            enable_statistics=False,
        )

    recorder = ValidationMetricRecorder(metadata)
    if isinstance(strategy_callback, LearnableGlobalDropoutCallback):
        recorder.rate_callback = strategy_callback

    callbacks = []
    if strategy_callback is not None:
        callbacks.append(strategy_callback)
    callbacks.extend((EarlyStopping(patience=10), recorder))
    return callbacks


def make_config(
    model_name,
    dataset_name,
    num_features,
    input_len,
    output_len,
    gpu_id,
    experiment,
    num_epochs,
):
    model_class, model_config, use_timestamps = get_model_config(
        model_name, input_len, output_len, num_features, dataset_name
    )
    variant = experiment["variant"]
    config_name = experiment["config_name"]
    seed = experiment["seed"]

    hyperparameters = {
        key: value
        for key, value in experiment.items()
        if key not in {"variant", "config_name", "seed"}
    }
    metadata = {
        "model": model_name,
        "model_class": model_class.__name__,
        "dataset": dataset_name,
        "input_len": input_len,
        "output_len": output_len,
        "num_epochs": num_epochs,
        "variant": variant,
        "config_name": config_name,
        "seed": seed,
        "hyperparameters": hyperparameters,
    }
    callbacks = make_callbacks(variant, metadata, experiment)
    batch_size = get_batch_size(model_name, dataset_name, output_len)

    # Fixed candidates are selected using validation only. Test evaluation is
    # launched later for the selected p* checkpoint.
    eval_after_train = variant != "fixed"
    config = BasicTSForecastingConfig(
        model=model_class,
        model_config=model_config,
        dataset_name=dataset_name,
        input_len=input_len,
        output_len=output_len,
        use_timestamps=use_timestamps,
        use_clean_targets=True,
        gpus=str(gpu_id),
        num_epochs=num_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        seed=seed,
        target_metric="MSE",
        eval_after_train=eval_after_train,
        deterministic=False,
        cudnn_benchmark=False,
        cudnn_determinstic=False,
        ckpt_save_dir=get_ckpt_save_dir(
            model_class,
            dataset_name,
            input_len,
            output_len,
            variant,
            config_name,
            seed,
        ),
        train_data_num_workers=DATA_NUM_WORKERS,
        val_data_num_workers=DATA_NUM_WORKERS,
        test_data_num_workers=DATA_NUM_WORKERS,
        train_data_pin_memory=True,
        val_data_pin_memory=True,
        test_data_pin_memory=True,
    )
    return config


def result_dir(config):
    return os.path.join(config.ckpt_save_dir, config.md5)


def run_experiment(
    model_name,
    dataset_name,
    num_features,
    input_len,
    output_len,
    gpu_id,
    experiment,
    num_epochs,
):
    config = make_config(
        model_name,
        dataset_name,
        num_features,
        input_len,
        output_len,
        gpu_id,
        experiment,
        num_epochs,
    )
    run_dir = result_dir(config)
    validation_path = os.path.join(run_dir, "validation_metrics.json")
    test_path = os.path.join(run_dir, "test_metrics.json")
    needs_test = experiment["variant"] != "fixed"

    if os.path.exists(validation_path) and (
        not needs_test or os.path.exists(test_path)
    ):
        print(f"    [Skip] Existing result: {run_dir}")
    else:
        BasicTSLauncher.launch_training(config)

    if not os.path.exists(validation_path):
        return None
    with open(validation_path, encoding="utf-8") as file:
        record = json.load(file)
    record["result_dir"] = run_dir
    return {"record": record, "config": config}


def select_fixed_rate(fixed_artifacts, seeds):
    values_by_rate = defaultdict(list)
    artifacts_by_rate_seed = {}
    for artifact in fixed_artifacts:
        if artifact is None:
            continue
        record = artifact["record"]
        rate = float(record["hyperparameters"]["fixed_p"])
        seed = int(record["seed"])
        values_by_rate[rate].append(float(record["validation_value"]))
        artifacts_by_rate_seed[(rate, seed)] = artifact

    complete_rates = {
        rate: values
        for rate, values in values_by_rate.items()
        if len(values) == len(seeds)
    }
    if not complete_rates:
        return None, [], artifacts_by_rate_seed

    curve = [
        {
            "p": rate,
            "validation_mse_mean": statistics.mean(values),
            "validation_mse_by_seed": values,
        }
        for rate, values in sorted(complete_rates.items())
    ]
    best = min(curve, key=lambda item: (item["validation_mse_mean"], item["p"]))
    return best["p"], curve, artifacts_by_rate_seed


def evaluate_selected_fixed(best_rate, artifacts_by_rate_seed, seeds, gpu_id):
    if best_rate is None:
        print("    [Fixed selection] No rate completed for every seed; skipping test.")
        return

    print(f"    [Fixed selection] Validation-selected p*={best_rate:.2f}")
    for seed in seeds:
        artifact = artifacts_by_rate_seed.get((best_rate, seed))
        if artifact is None:
            print(f"    [Fixed selection] Missing seed {seed}; skipping.")
            continue

        config = artifact["config"]
        run_dir = artifact["record"]["result_dir"]
        test_path = os.path.join(run_dir, "test_metrics.json")
        if os.path.exists(test_path):
            print(f"    [Skip] Selected fixed test exists: {test_path}")
            continue

        checkpoints = glob.glob(os.path.join(run_dir, "*_best_val_MSE.pt"))
        if len(checkpoints) != 1:
            print(
                "    [Fixed selection] Expected one best checkpoint in "
                f"{run_dir}, found {len(checkpoints)}; skipping."
            )
            continue
        BasicTSLauncher.launch_evaluation(
            config,
            checkpoints[0],
            gpus=str(gpu_id),
            batch_size=config.test_batch_size,
        )


def worker_task(
    gpu_queue,
    model_name,
    dataset_name,
    num_features,
    input_len,
    output_len,
    seeds,
    num_epochs,
):
    gpu_id = None
    task_id = f"{model_name} {dataset_name} (L={input_len}, H={output_len})"
    try:
        gpu_id = gpu_queue.get()
        print(f"[Start] {task_id} on GPU {gpu_id}")
        experiments = build_experiments(seeds)
        fixed_artifacts = []

        for index, experiment in enumerate(experiments, start=1):
            label = (
                f"{experiment['variant']}/{experiment['config_name']}/"
                f"seed_{experiment['seed']}"
            )
            print(f"  [Exp {index}/{len(experiments)}] {label}")
            try:
                artifact = run_experiment(
                    model_name,
                    dataset_name,
                    num_features,
                    input_len,
                    output_len,
                    str(gpu_id),
                    experiment,
                    num_epochs,
                )
                if experiment["variant"] == "fixed":
                    fixed_artifacts.append(artifact)
            except Exception as error:
                print(f"    [Error] {task_id} {label}: {error}")
                import traceback

                traceback.print_exc()

        best_rate, _, by_rate_seed = select_fixed_rate(fixed_artifacts, seeds)
        evaluate_selected_fixed(best_rate, by_rate_seed, seeds, gpu_id)
    except Exception as error:
        print(f"[Error] {task_id}: {error}")
        import traceback

        traceback.print_exc()
    finally:
        if gpu_id is not None:
            gpu_queue.put(gpu_id)
            print(f"[Done] {task_id} released GPU {gpu_id}")


def read_result_record(path):
    with open(path, encoding="utf-8") as file:
        record = json.load(file)
    run_dir = os.path.dirname(path)
    test_path = os.path.join(run_dir, "test_metrics.json")
    if os.path.exists(test_path):
        with open(test_path, encoding="utf-8") as file:
            record["test_metrics"] = json.load(file).get("overall", {})
    record["result_dir"] = run_dir
    return record


def metric_summary(records, metric):
    values = [
        record["test_metrics"][metric]
        for record in records
        if metric in record.get("test_metrics", {})
    ]
    if not values:
        return None
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
        "values": values,
    }


def write_summary():
    paths = glob.glob(
        os.path.join(CHECKPOINT_ROOT, "**", "validation_metrics.json"),
        recursive=True,
    )
    records = [read_result_record(path) for path in paths]
    grouped = defaultdict(list)
    for record in records:
        key = (
            record["model"],
            record["model_class"],
            record["dataset"],
            record["input_len"],
            record["output_len"],
            record.get("num_epochs", NUM_EPOCHS),
        )
        grouped[key].append(record)

    tasks = []
    for key, task_records in sorted(grouped.items()):
        fixed_records = [
            record for record in task_records if record["variant"] == "fixed"
        ]
        seeds = sorted({int(record["seed"]) for record in fixed_records})
        values_by_rate = defaultdict(list)
        for record in fixed_records:
            rate = float(record["hyperparameters"]["fixed_p"])
            values_by_rate[rate].append(float(record["validation_value"]))

        fixed_curve = [
            {
                "p": rate,
                "validation_mse_mean": statistics.mean(values),
                "validation_mse_by_seed": values,
            }
            for rate, values in sorted(values_by_rate.items())
            if len(values) == len(seeds)
        ]
        best_rate = None
        if fixed_curve:
            best_rate = min(
                fixed_curve,
                key=lambda item: (item["validation_mse_mean"], item["p"]),
            )["p"]

        selected = {
            "original": [
                record for record in task_records if record["variant"] == "original"
            ],
            "best_fixed": [
                record
                for record in fixed_records
                if best_rate is not None
                and math.isclose(
                    float(record["hyperparameters"]["fixed_p"]), best_rate
                )
            ],
            "learnable_global": [
                record
                for record in task_records
                if record["variant"] == "learnable_global"
            ],
            "dropoutts": [
                record for record in task_records if record["variant"] == "dropoutts"
            ],
        }
        aggregate = {
            variant: {
                "MSE": metric_summary(variant_records, "MSE"),
                "MAE": metric_summary(variant_records, "MAE"),
            }
            for variant, variant_records in selected.items()
        }

        tasks.append(
            {
                "model": key[0],
                "model_class": key[1],
                "dataset": key[2],
                "input_len": key[3],
                "output_len": key[4],
                "num_epochs": key[5],
                "fixed_selection": {
                    "criterion": "lowest mean validation MSE across completed seeds",
                    "selected_p": best_rate,
                    "curve": fixed_curve,
                },
                "aggregate_test_metrics": aggregate,
                "selected_records": selected,
            }
        )

    os.makedirs(CHECKPOINT_ROOT, exist_ok=True)
    summary_path = os.path.join(CHECKPOINT_ROOT, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "selection_uses_test_data": False,
                "fixed_rate_grid": FIXED_RATES,
                "tasks": tasks,
            },
            file,
            indent=2,
        )
    print(f"Summary written to: {summary_path}")


def validate_datasets(tasks):
    missing = []
    for dataset_name in sorted({task[1] for task in tasks}):
        dataset_dir = os.path.join(PROJECT_ROOT, "datasets", dataset_name)
        required = (
            "train_data.npy",
            "val_data.npy",
            "test_data.npy",
        )
        absent = [
            filename
            for filename in required
            if not os.path.exists(os.path.join(dataset_dir, filename))
        ]
        if absent:
            missing.append(f"{dataset_name}: {', '.join(absent)}")
    if missing:
        raise FileNotFoundError(
            "Missing required dataset files:\n  " + "\n  ".join(missing)
        )


def parse_int_list(value):
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return values


def print_plan(tasks, seeds, all_horizons):
    training_runs_per_task = len(build_experiments(seeds))
    total_training_runs = len(tasks) * training_runs_per_task
    print(
        f"Scope: {len(tasks)} model-dataset-horizon tasks; "
        f"{training_runs_per_task} training runs per task; "
        f"{total_training_runs} training runs total."
    )
    print(
        f"Horizon mode: {'all paper horizons' if all_horizons else 'representative horizon'}; "
        f"seeds={list(seeds)}."
    )
    print(
        "Each task runs Original + 11 Fixed candidates + Learnable Global + "
        "DropoutTS. Only validation-selected Fixed p* is tested."
    )
    for task in tasks:
        print(
            f"  {task[0]:11s} {task[1]:22s} "
            f"L={task[3]:3d} H={task[4]:3d}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run the representative fixed/global/adaptive dropout comparison."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact task matrix without launching training.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Rebuild summary.json from completed runs and exit.",
    )
    parser.add_argument(
        "--all-horizons",
        action="store_true",
        help="Use all four paper horizons instead of one representative horizon.",
    )
    parser.add_argument(
        "--gpus",
        type=parse_int_list,
        default=parse_int_list("0,1,2,3,4,5,6,7"),
        help="Comma-separated physical GPU ids (default: 0,1,2,3,4,5,6,7).",
    )
    parser.add_argument(
        "--jobs-per-gpu",
        type=int,
        default=2,
        help="Concurrent task workers per GPU (default: 2).",
    )
    parser.add_argument(
        "--seeds",
        type=parse_int_list,
        default=parse_int_list("42"),
        help="Comma-separated paired seeds (default: 42).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help=f"Maximum training epochs (default: {NUM_EPOCHS}).",
    )
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    if args.summary_only:
        write_summary()
        return
    if args.jobs_per_gpu <= 0:
        parser.error("--jobs-per-gpu must be positive.")
    if args.epochs <= 0:
        parser.error("--epochs must be positive.")

    tasks = build_tasks(all_horizons=args.all_horizons)
    validate_datasets(tasks)
    print_plan(tasks, args.seeds, args.all_horizons)
    if args.dry_run:
        return

    gpu_queue = Queue()
    for gpu_id in args.gpus:
        for _ in range(args.jobs_per_gpu):
            gpu_queue.put(gpu_id)

    processes = []
    max_concurrent = len(args.gpus) * args.jobs_per_gpu
    print(
        f"Scheduling on GPUs {list(args.gpus)} with up to "
        f"{max_concurrent} concurrent workers."
    )
    for task in tasks:
        process = Process(
            target=worker_task,
            args=(gpu_queue, *task, args.seeds, args.epochs),
        )
        process.start()
        processes.append(process)
        time.sleep(0.1)

    print(f"Scheduled {len(processes)} task workers.")
    try:
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        print("Interrupted; terminating child workers.")
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()
        raise

    failed = [process.pid for process in processes if process.exitcode != 0]
    if failed:
        print(f"Workers with non-zero exit status: {failed}")
    write_summary()
    print("All task workers finished.")


if __name__ == "__main__":
    main()
