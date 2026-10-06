import sys
import os
import time
from itertools import product
from multiprocessing import Process, Queue

# Add src directory
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, 'src')
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
from basicts.runners.callback import EarlyStopping, DropoutTSCallback
from basicts import BasicTSLauncher

# --- Global Configurations ---
AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
JOBS_PER_GPU = 2
NUM_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
LARGE_BATCH_SIZE = 16
TINY_BATCH_SIZE = 1
DATA_NUM_WORKERS = 4
MODELS = [
    "PatchTST",
    "Crossformer",
    "Informer",
    "iTransformer",
    "TimeMixer",
    "TimesNet",
    "TimeFilter",
    "WPMixer",
    "MultiPatchFormer",
]
DATASETS = [
    ("SyntheticTS_noise0.1", 1),
    ("SyntheticTS_noise0.3", 1),
    ("SyntheticTS_noise0.5", 1),
    ("SyntheticTS_noise0.7", 1),
    ("SyntheticTS_noise0.9", 1),
    ("ETTh1", 7),
    ("ETTh2", 7),
    ("ETTm1", 7),
    ("ETTm2", 7),
    ("Electricity", 321),
    ("Weather", 21),
    ("Illness", 7),
    ("ExchangeRate", 8),
    ("ECG", 12),
]

RUN_RAW_BASELINE = True
RUN_DROPOUT_TS = True

# Length settings
DATASET_CONFIGS = {
    "Illness": {"input_lens": [24], "output_lens": [24, 36, 48, 60]},
    "ExchangeRate": {"input_lens": [96], "output_lens": [96, 192, 336, 720]},
    "ECG": {"input_lens": [96], "output_lens": [36, 72, 144, 288]},
    "default": {"input_lens": [96], "output_lens": [96, 192, 336, 720]}
}

# Hyperparameters
HPARAMS = {
    "p_min": [0.05],
    "p_max": [0.5],
    "init_alpha": [10.0],
    "init_sensitivity": [1.0],
    "sparsity_weight": [0.0],
    "use_gate": [False],
    "train_sample_ratio": [1.0],
}
USE_CLEAN_TARGETS = True


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
    experiments = []

    if RUN_RAW_BASELINE:
        experiments.append({
            "name": "raw",
            "enable_dropout_ts": False,
            "train_sample_ratio": 1.0,
        })

    if RUN_DROPOUT_TS:
        param_combinations = product(
            HPARAMS["p_min"],
            HPARAMS["p_max"],
            HPARAMS["init_alpha"],
            HPARAMS["init_sensitivity"],
            HPARAMS["sparsity_weight"],
            HPARAMS["use_gate"],
            HPARAMS["train_sample_ratio"],
        )
        for p_min, p_max, alpha, sens, sp_weight, use_gate, train_ratio in param_combinations:
            if p_max <= p_min:
                continue
            experiments.append({
                "name": f"dt_sens={sens}_sp={sp_weight}_gate={use_gate}_tr={train_ratio}",
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


def get_ckpt_save_dir(model_class, dataset_name, input_len, output_len):
    return os.path.join(
        "checkpoints",
        model_class.__name__,
        f"{dataset_name}_{NUM_EPOCHS}_{input_len}_{output_len}",
    )


def run_experiment(model_name, dataset_name, num_features, input_len, output_len, gpu_id, **kwargs):
    """Setup and launch a single experiment."""
    model_class, model_config, use_timestamps = get_model_config(
        model_name, input_len, output_len, num_features, dataset_name
    )
    
    callbacks = [EarlyStopping(patience=10)]
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
    
    batch_size = kwargs.get(
        "batch_size", get_batch_size(model_name, dataset_name, output_len)
    )
    cfg = BasicTSForecastingConfig(
        model=model_class, model_config=model_config,
        dataset_name=dataset_name, input_len=input_len, output_len=output_len,
        use_timestamps=use_timestamps, use_clean_targets=USE_CLEAN_TARGETS,
        train_sample_ratio=kwargs.get('train_sample_ratio', 1.0),
        gpus=gpu_id, num_epochs=NUM_EPOCHS, batch_size=batch_size,
        callbacks=callbacks, seed=42,
        ckpt_save_dir=get_ckpt_save_dir(model_class, dataset_name, input_len, output_len),
        train_data_num_workers=DATA_NUM_WORKERS,
        val_data_num_workers=DATA_NUM_WORKERS,
        test_data_num_workers=DATA_NUM_WORKERS,
        train_data_pin_memory=True,
        val_data_pin_memory=True,
        test_data_pin_memory=True,
    )
    metrics_path = os.path.join(cfg.ckpt_save_dir, cfg.md5, "test_metrics.json")
    if os.path.exists(metrics_path):
        print(f"    [Skip] Existing metrics at {metrics_path}")
        return
    BasicTSLauncher.launch_training(cfg)


def worker_task(gpu_queue, model_name, dataset_name, num_features, input_len, output_len):
    """Worker process: Acquires GPU -> Runs experiments -> Releases GPU."""
    gpu_id = None
    task_id = f"{model_name} {dataset_name} ({input_len}->{output_len})"

    try:
        gpu_id = gpu_queue.get()
        print(f"[Start] {task_id} on GPU {gpu_id}")

        experiments = build_experiments()
        total_exps = len(experiments)
        print(f"  -> Check {total_exps} experiments on this GPU.")

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


if __name__ == "__main__":
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

    for model_name in MODELS:
        for dataset_name, num_features in DATASETS:
            config = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS["default"])

            for input_len in config["input_lens"]:
                for output_len in config["output_lens"]:
                    p = Process(
                        target=worker_task,
                        args=(gpu_queue, model_name, dataset_name, num_features, input_len, output_len)
                    )
                    p.start()
                    processes.append(p)
                    time.sleep(0.1)

    print(f"Scheduled {len(processes)} task workers.")
    
    for p in processes:
        p.join()

    print("All task workers finished.")
