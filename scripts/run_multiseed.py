import argparse
import glob
import json
import math
import sys
import os
import statistics
import time
from itertools import product
from multiprocessing import Process, Queue

from scipy.stats import ttest_rel

# Add src directory (the repository root is one level above this script)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(project_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import Models
from basicts.models.Crossformer import Crossformer, CrossformerConfig
from basicts.models.Informer import Informer, InformerConfig
from basicts.models.iTransformer import iTransformerForForecasting, iTransformerConfig
from basicts.models.PatchTST import PatchTSTForForecasting, PatchTSTConfig
from basicts.models.TimeMixer import TimeMixerForForecasting, TimeMixerConfig
from basicts.models.TimesNet import TimesNetForForecasting, TimesNetConfig
from basicts.models.WPMixer import WPMixerForForecasting, WPMixerConfig
from basicts.models.TimeFilter import TimeFilterForForecasting, TimeFilterConfig
from basicts.models.MultiPatchFormer import MultiPatchFormerForForecasting, MultiPatchFormerConfig
from basicts.configs import BasicTSForecastingConfig
from basicts.runners.callback import BasicTSCallback, EarlyStopping, DropoutTSCallback
from basicts import BasicTSLauncher

# --- Global Configurations ---
AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
JOBS_PER_GPU = 2
NUM_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
LARGE_BATCH_SIZE = 16
TINY_BATCH_SIZE = 1
DATA_NUM_WORKERS = 4
EXPERIMENT_GROUP = "representative_hparam_search"
SEEDS = [2022, 2023, 2024, 2025, 2026]
MODELS = [
    "TimeFilter",
    "PatchTST",
    "TimeMixer",
    "Informer",
]
DATASETS = [
    ("ETTh1", 7),
    ("Illness", 7),
    ("ExchangeRate", 8),
]

RUN_RAW_BASELINE = True
RUN_DROPOUT_TS = True

# Length settings
DATASET_CONFIGS = {
    "ETTh1": {"input_lens": [96], "output_lens": [96]},
    "Illness": {"input_lens": [24], "output_lens": [24]},
    "ExchangeRate": {"input_lens": [96], "output_lens": [96]},
}

# DropoutTS hyperparameter search space.
DROPOUT_TS_CONFIG = {
    "p_min": [0.05],
    "p_max": [0.5],
    "init_alpha": [10.0],
    "init_sensitivity": [1.0],
    "sparsity_weight": [0.0],
    "use_gate": [False],
    "train_sample_ratio": [1.0],
}
USE_CLEAN_TARGETS = True


class ValidationMetricRecorder(BasicTSCallback):
    """Persist each run's best validation score for later analysis."""

    def __init__(self, metadata):
        self.metadata = metadata

    def on_train_end(self, runner, *args, **kwargs):
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
        record_path = os.path.join(runner.ckpt_save_dir, "validation_metrics.json")
        with open(record_path, "w", encoding="utf-8") as file:
            json.dump(record, file, indent=2)


def get_timestamp_sizes(dataset_name: str):
    """Helper to get timestamp features."""
    if dataset_name in ["ETTh1", "ETTh2"]: return [24, 7, 31, 366]
    if dataset_name in ["ETTm1", "ETTm2", "SyntheticTS"]: return [96, 7, 31, 366]
    if dataset_name.startswith("SyntheticTS"): return [96, 7, 31, 366]
    return [60, 7, 31, 366]


def get_model_config(model_name, input_len, output_len, num_features, dataset_name):
    """Factory to create model class and config using explicit keywords."""
    
    if model_name == "Crossformer":
        cfg = CrossformerConfig(
            input_len=input_len, 
            output_len=output_len, 
            num_features=num_features
        )
        return Crossformer, cfg, False

    elif model_name == "Informer":
        cfg = InformerConfig(
            input_len=input_len, 
            output_len=output_len, 
            label_len=output_len // 2, 
            num_features=num_features, 
            use_timestamps=True, 
            timestamp_sizes=get_timestamp_sizes(dataset_name)
        )
        return Informer, cfg, True

    elif model_name == "iTransformer":
        cfg = iTransformerConfig(
            input_len=input_len, 
            output_len=output_len, 
            num_features=num_features
        )
        return iTransformerForForecasting, cfg, False

    elif model_name == "PatchTST":
        cfg = PatchTSTConfig(
            input_len=input_len, 
            output_len=output_len, 
            num_features=num_features
        )
        return PatchTSTForForecasting, cfg, False

    elif model_name == "TimeMixer":
        cfg = TimeMixerConfig(
            input_len=input_len, 
            output_len=output_len, 
            num_features=num_features
        )
        return TimeMixerForForecasting, cfg, False

    elif model_name == "TimesNet":
        cfg = TimesNetConfig(
            input_len=input_len, 
            output_len=output_len, 
            num_features=num_features, 
            use_timestamps=True, 
            timestamp_sizes=get_timestamp_sizes(dataset_name)
        )
        return TimesNetForForecasting, cfg, True

    elif model_name == "WPMixer":
        cfg = WPMixerConfig(
            input_len=input_len,
            output_len=output_len,
            num_features=num_features,
        )
        return WPMixerForForecasting, cfg, False

    elif model_name == "TimeFilter":
        cfg = TimeFilterConfig(
            input_len=input_len,
            output_len=output_len,
            num_features=num_features,
        )
        # TimeFilter does not use timestamps in the reference implementation.
        return TimeFilterForForecasting, cfg, False

    elif model_name == "MultiPatchFormer":
        cfg = MultiPatchFormerConfig(
            input_len=input_len,
            output_len=output_len,
            num_features=num_features,
        )
        return MultiPatchFormerForForecasting, cfg, False

    else:
        raise ValueError(f"Unknown model: {model_name}")


def build_experiments():
    """Build Raw and the full DropoutTS grid for every seed."""
    experiments = []

    for seed in SEEDS:
        if RUN_RAW_BASELINE:
            experiments.append({
                "name": f"raw/seed_{seed}",
                "variant": "raw",
                "config_name": "default",
                "seed": seed,
                "enable_dropout_ts": False,
                "train_sample_ratio": 1.0,
            })

        if RUN_DROPOUT_TS:
            param_combinations = product(
                DROPOUT_TS_CONFIG["p_min"],
                DROPOUT_TS_CONFIG["p_max"],
                DROPOUT_TS_CONFIG["init_alpha"],
                DROPOUT_TS_CONFIG["init_sensitivity"],
                DROPOUT_TS_CONFIG["sparsity_weight"],
                DROPOUT_TS_CONFIG["use_gate"],
                DROPOUT_TS_CONFIG["train_sample_ratio"],
            )
            for p_min, p_max, alpha, sens, sp_weight, use_gate, train_ratio in param_combinations:
                if p_max <= p_min:
                    continue

                config_name = (
                    f"pmin_{p_min:g}_pmax_{p_max:g}_alpha_{alpha:g}_"
                    f"sens_{sens:g}_sp_{sp_weight:g}_gate_{str(use_gate).lower()}_"
                    f"tr_{train_ratio:g}"
                ).replace(".", "p")
                experiments.append({
                    "name": f"dropoutts/{config_name}/seed_{seed}",
                    "variant": "dropoutts",
                    "config_name": config_name,
                    "seed": seed,
                    "enable_dropout_ts": True,
                    "p_min": p_min,
                    "p_max": p_max,
                    "init_alpha": alpha,
                    "init_sensitivity": sens,
                    "sparsity_weight": sp_weight,
                    "use_gate": use_gate,
                    "train_sample_ratio": train_ratio,
                })

    return experiments


def get_batch_size(model_name, dataset_name, output_len):
    if model_name == "TimeFilter" and dataset_name in {"Electricity", "ECG"}:
        return TINY_BATCH_SIZE
    if model_name == "MultiPatchFormer" or dataset_name in {"Electricity", "ECG"}:
        return LARGE_BATCH_SIZE
    if output_len >= 720:
        return LARGE_BATCH_SIZE
    return DEFAULT_BATCH_SIZE


def get_ckpt_save_dir(
    model_class, dataset_name, input_len, output_len, variant, config_name, seed
):
    return os.path.join(
        "checkpoints",
        EXPERIMENT_GROUP,
        model_class.__name__,
        f"{dataset_name}_L{input_len}_H{output_len}",
        variant,
        config_name,
        f"seed_{seed}",
    )


def run_experiment(
    model_name,
    dataset_name,
    num_features,
    input_len,
    output_len,
    gpu_id,
    experiment_name,
    variant,
    config_name,
    seed,
    **kwargs,
):
    """Setup and launch a single experiment."""
    model_class, model_config, use_timestamps = get_model_config(
        model_name, input_len, output_len, num_features, dataset_name
    )

    search_metadata = {
        "model": model_name,
        "model_class": model_class.__name__,
        "dataset": dataset_name,
        "input_len": input_len,
        "output_len": output_len,
        "variant": variant,
        "config_name": config_name,
        "seed": seed,
        "hyperparameters": {
            key: kwargs[key]
            for key in DROPOUT_TS_CONFIG
            if key in kwargs
        },
    }
    callbacks = [
        EarlyStopping(patience=10),
        ValidationMetricRecorder(search_metadata),
    ]
    if kwargs.get('enable_dropout_ts'):
        callbacks.insert(0, DropoutTSCallback(
            p_min=kwargs['p_min'], 
            p_max=kwargs['p_max'],
            init_alpha=kwargs['init_alpha'], 
            init_sensitivity=kwargs['init_sensitivity'],
            sparsity_weight=kwargs['sparsity_weight'],
            enable_visualization=False, 
            enable_statistics=False,
            use_gate=kwargs['use_gate']
        ))
    
    batch_size = get_batch_size(model_name, dataset_name, output_len)
    cfg = BasicTSForecastingConfig(
        model=model_class, model_config=model_config,
        dataset_name=dataset_name, input_len=input_len, output_len=output_len,
        use_timestamps=use_timestamps, use_clean_targets=USE_CLEAN_TARGETS,
        train_sample_ratio=kwargs.get('train_sample_ratio', 1.0),
        gpus=gpu_id, num_epochs=NUM_EPOCHS, batch_size=batch_size,
        callbacks=callbacks, seed=seed, target_metric="MSE",
        # Informer ProbAttention and TimeFilter's MoE use CUDA cumsum kernels
        # that PyTorch cannot execute under strict deterministic algorithms.
        # Seeds remain fixed for paired repeats; benchmark selection is disabled.
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
    metrics_path = os.path.join(cfg.ckpt_save_dir, cfg.md5, "test_metrics.json")
    validation_path = os.path.join(
        cfg.ckpt_save_dir, cfg.md5, "validation_metrics.json"
    )
    if os.path.exists(metrics_path) and os.path.exists(validation_path):
        print(f"    [Skip] {experiment_name}: existing metrics at {metrics_path}")
    else:
        BasicTSLauncher.launch_training(cfg)

    if not os.path.exists(validation_path):
        return None
    with open(validation_path, encoding="utf-8") as file:
        record = json.load(file)
    record["result_dir"] = os.path.dirname(validation_path)
    return record


def metric_stats(values):
    """Return mean, sample standard deviation, and sample variance."""
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
        "variance": statistics.variance(values) if len(values) > 1 else None,
    }


def load_result_records(metric_paths):
    records_by_seed = {}
    for metric_path in metric_paths:
        with open(metric_path, encoding="utf-8") as file:
            record = json.load(file)

        test_metrics_path = os.path.join(os.path.dirname(metric_path), "test_metrics.json")
        if not os.path.exists(test_metrics_path):
            continue
        with open(test_metrics_path, encoding="utf-8") as file:
            record["test_metrics"] = json.load(file).get("overall", {})
        record["result_dir"] = os.path.dirname(metric_path)
        record_mtime = os.path.getmtime(test_metrics_path)
        previous = records_by_seed.get(record["seed"])
        if previous is None or record_mtime > previous["_mtime"]:
            record["_mtime"] = record_mtime
            records_by_seed[record["seed"]] = record

    records = []
    for record in records_by_seed.values():
        record.pop("_mtime", None)
        records.append(record)
    return sorted(records, key=lambda record: record["seed"])


def summarize_repeats(records):
    if not records:
        return None

    metric_names = sorted(
        set.intersection(*(set(record["test_metrics"]) for record in records))
    )
    return {
        "n": len(records),
        "seeds": [record["seed"] for record in records],
        "validation_mse": metric_stats(
            [record["validation_value"] for record in records]
        ),
        "test_metrics": {
            metric_name: metric_stats(
                [record["test_metrics"][metric_name] for record in records]
            )
            for metric_name in metric_names
        },
        "result_dirs": [record["result_dir"] for record in records],
    }


def summarize_paired_mse(raw_records, dropoutts_records):
    """Compare matched test MSE values across the common seeds."""
    raw_by_seed = {record["seed"]: record for record in raw_records}
    dropoutts_by_seed = {record["seed"]: record for record in dropoutts_records}
    paired_seeds = sorted(set(raw_by_seed) & set(dropoutts_by_seed))
    if not paired_seeds:
        return None

    raw_mse = [raw_by_seed[seed]["test_metrics"]["MSE"] for seed in paired_seeds]
    dropoutts_mse = [
        dropoutts_by_seed[seed]["test_metrics"]["MSE"] for seed in paired_seeds
    ]
    wins = sum(dt < raw and not math.isclose(dt, raw) for raw, dt in zip(raw_mse, dropoutts_mse))
    ties = sum(math.isclose(dt, raw) for raw, dt in zip(raw_mse, dropoutts_mse))
    losses = len(paired_seeds) - wins - ties
    p_value = None
    if len(paired_seeds) > 1:
        test_result = ttest_rel(raw_mse, dropoutts_mse)
        if not math.isnan(test_result.pvalue):
            p_value = float(test_result.pvalue)

    raw_mean = statistics.mean(raw_mse)
    dropoutts_mean = statistics.mean(dropoutts_mse)
    return {
        "n": len(paired_seeds),
        "seeds": paired_seeds,
        "relative_gain_percent": 100.0 * (raw_mean - dropoutts_mean) / raw_mean,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "two_sided_paired_t_test_p": p_value,
    }


def write_all_results_summary():
    """Aggregate Raw and every DropoutTS configuration without selecting one."""
    search_root = os.path.join("checkpoints", EXPERIMENT_GROUP)
    tasks = []
    task_dirs = sorted(glob.glob(os.path.join(search_root, "*", "*_L*_H*")))
    for task_dir in task_dirs:
        raw_records = load_result_records(
            glob.glob(
                os.path.join(
                    task_dir, "raw", "default", "seed_*", "*", "validation_metrics.json"
                )
            )
        )
        configurations = []
        config_dirs = sorted(glob.glob(os.path.join(task_dir, "dropoutts", "*")))
        for config_dir in config_dirs:
            if not os.path.isdir(config_dir):
                continue
            dropoutts_records = load_result_records(
                glob.glob(
                    os.path.join(
                        config_dir,
                        "seed_*",
                        "*",
                        "validation_metrics.json",
                    )
                )
            )
            if not dropoutts_records:
                continue
            configurations.append({
                "config_name": os.path.basename(config_dir),
                "hyperparameters": dropoutts_records[0]["hyperparameters"],
                "dropoutts": summarize_repeats(dropoutts_records),
                "paired_test_mse": summarize_paired_mse(
                    raw_records, dropoutts_records
                ),
            })

        available_records = raw_records or [
            record
            for config in config_dirs
            for record in load_result_records(
                glob.glob(
                    os.path.join(
                        config, "seed_*", "*", "validation_metrics.json"
                    )
                )
            )
        ]
        if not available_records:
            continue
        metadata = available_records[0]
        tasks.append({
            "model": metadata["model"],
            "model_class": metadata["model_class"],
            "dataset": metadata["dataset"],
            "input_len": metadata["input_len"],
            "output_len": metadata["output_len"],
            "raw": summarize_repeats(raw_records),
            "dropoutts_configurations": configurations,
        })

    if not tasks:
        print("No completed results found; result summary was not written.")
        return

    summary = {
        "selection_performed": False,
        "seeds": SEEDS,
        "tasks": tasks,
    }
    os.makedirs(search_root, exist_ok=True)
    summary_path = os.path.join(search_root, "all_results.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(f"All five-seed results written to: {summary_path}")


def worker_task(gpu_queue, model_name, dataset_name, num_features, input_len, output_len):
    """Worker process: Acquires GPU -> Runs experiments -> Releases GPU."""
    gpu_id = None
    task_id = f"{model_name} {dataset_name} ({input_len}->{output_len})"

    try:
        gpu_id = gpu_queue.get()
        print(f"[Start] {task_id} on GPU {gpu_id}")

        experiments = build_experiments()
        total_exps = len(experiments)
        print(f"  -> Run {total_exps} raw/grid experiments on this GPU.")

        for idx, exp in enumerate(experiments):
            exp_name = exp["name"]
            print(f"    [Exp {idx+1}/{total_exps}] {exp_name}")
            try:
                run_kwargs = {k: v for k, v in exp.items() if k != "name"}
                run_experiment(
                    model_name,
                    dataset_name,
                    num_features,
                    input_len,
                    output_len,
                    str(gpu_id),
                    experiment_name=exp_name,
                    **run_kwargs,
                )
            except Exception as e:
                print(f"    [Error] {task_id} {exp_name} failed: {e}")
                import traceback; traceback.print_exc()

    except Exception as e:
        print(f"[Error] {task_id} failed: {e}")
        import traceback; traceback.print_exc()
    finally:
        if gpu_id is not None:
            gpu_queue.put(gpu_id)
            print(f"[Done] {task_id} released GPU {gpu_id}")


def main(dry_run=False):
    tasks = []
    for model_name in MODELS:
        for dataset_name, num_features in DATASETS:
            config = DATASET_CONFIGS[dataset_name]
            for input_len in config["input_lens"]:
                for output_len in config["output_lens"]:
                    tasks.append(
                        (model_name, dataset_name, num_features, input_len, output_len)
                    )

    experiments_per_task = len(build_experiments())
    raw_runs_per_task = len(SEEDS) if RUN_RAW_BASELINE else 0
    dropout_configs_per_seed = (
        (experiments_per_task - raw_runs_per_task) // len(SEEDS)
        if RUN_DROPOUT_TS else 0
    )
    total_runs = len(tasks) * experiments_per_task
    print(
        f"Representative hyperparameter search: {len(tasks)} model-dataset tasks, "
        f"{experiments_per_task} runs per task, {total_runs} runs total."
    )
    print(
        f"Seeds: {SEEDS}; each seed runs one Raw baseline and all "
        f"{dropout_configs_per_seed} "
        "DropoutTS configurations."
    )

    if dry_run:
        for model_name, dataset_name, num_features, input_len, output_len in tasks:
            model_class, _, _ = get_model_config(
                model_name, input_len, output_len, num_features, dataset_name
            )
            print(
                f"  {model_name}: {dataset_name} L={input_len}, H={output_len} "
                f"[{model_class.__name__}]"
            )
        print("Experiments per task:")
        for experiment in build_experiments():
            print(f"  {experiment['name']}")
        return

    gpu_queue = Queue()
    for gpu_id in AVAILABLE_GPUS:
        for _ in range(JOBS_PER_GPU):
            gpu_queue.put(gpu_id)

    processes = []
    max_concurrent = len(AVAILABLE_GPUS) * JOBS_PER_GPU

    print(
        f"Scheduling tasks on GPUs: {AVAILABLE_GPUS} "
        f"(up to {max_concurrent} concurrent, JOBS_PER_GPU={JOBS_PER_GPU})"
    )

    for model_name, dataset_name, num_features, input_len, output_len in tasks:
        p = Process(
            target=worker_task,
            args=(
                gpu_queue,
                model_name,
                dataset_name,
                num_features,
                input_len,
                output_len,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.1)

    print(f"Scheduled {len(processes)} task workers.")
    
    for p in processes:
        p.join()

    print("All task workers finished.")
    write_all_results_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the representative forecasting hyperparameter search."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected tasks without launching training.",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
