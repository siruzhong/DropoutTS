import sys
import os
import time
from itertools import product
from multiprocessing import Process, Queue

# Add src directory (the repository root is one level above this script)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(project_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import Models
from basicts.models.Crossformer import Crossformer, CrossformerConfig
from basicts.models.DLinear import DLinear, DLinearConfig
from basicts.models.Informer import Informer, InformerConfig
from basicts.models.iTransformer import iTransformerForForecasting, iTransformerConfig
from basicts.models.PatchTST import PatchTSTForForecasting, PatchTSTConfig
from basicts.models.TimeMixer import TimeMixerForForecasting, TimeMixerConfig
from basicts.models.TimesNet import TimesNetForForecasting, TimesNetConfig
from basicts.configs import BasicTSForecastingConfig
from basicts.runners.callback import EarlyStopping, DropoutTSCallback, SelectiveLearning
from basicts import BasicTSLauncher

# --- Global Configurations ---
AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
MODELS = ["PatchTST", "Crossformer", "Informer", "iTransformer", "TimeMixer", "TimesNet"]
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
]

# Length settings
DATASET_CONFIGS = {
    "Illness": {"input_lens": [24], "output_lens": [60]},
    "default": {"input_lens": [96], "output_lens": [96, 192, 336, 720]}
}

# DropoutTS Hyperparameters
HPARAMS = {
    "p_min": [0.05],
    "p_max": [0.5],
    "init_alpha": [10.0],
    "init_sensitivity": [1.0],
    "sparsity_weight": [0.0],
    "use_gate": [False]
}

# Selective Learning Hyperparameters
SL_HPARAMS = {
    "r_u": [0.1],      # Uncertainty mask ratio
    "r_a": [0.1],      # Anomaly mask ratio (requires DLinear estimator)
}

# DLinear Estimator Config for Selective Learning (r_a mode)
# Set ESTIMATOR_CKPT_PATH to the path of pretrained DLinear checkpoint
# Run `python scripts/run_pretrain_estimator.py` first to get the checkpoint
ESTIMATOR_CKPT_PATH = "checkpoints/DLinear/Illness_100_24_60/116f128db0e156488b183076e3df50a8/DLinear_best_val_MAE.pt"

# Experiment Mode: "dropoutts", "sl", "dropoutts+sl", "all"
EXPERIMENT_MODE = "all"

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

    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_experiment(model_name, dataset_name, num_features, input_len, output_len, gpu_id, **kwargs):
    """Setup and launch a single experiment.
    
    Supports:
        - enable_dropout_ts: Enable DropoutTS callback
        - enable_sl: Enable Selective Learning callback
        - Both can be enabled simultaneously (DropoutTS + SL)
    """
    model_class, model_config, use_timestamps = get_model_config(
        model_name, input_len, output_len, num_features, dataset_name
    )
    
    callbacks = [EarlyStopping(patience=10)]
    
    # Add DropoutTS callback (should be first to compute adaptive rates before SL)
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
    
    # Add Selective Learning callback
    if kwargs.get('enable_sl'):
        sl_kwargs = {}
        if kwargs.get('r_u') is not None:
            sl_kwargs['r_u'] = kwargs['r_u']
        if kwargs.get('r_a') is not None:
            sl_kwargs['r_a'] = kwargs['r_a']
            # Create DLinear estimator instance
            estimator_config = DLinearConfig(
                input_len=input_len,
                output_len=output_len,
                num_features=num_features
            )
            sl_kwargs['estimator'] = DLinear(estimator_config)
            sl_kwargs['ckpt_path'] = kwargs.get('ckpt_path')
        callbacks.append(SelectiveLearning(**sl_kwargs))
    
    cfg = BasicTSForecastingConfig(
        model=model_class, model_config=model_config,
        dataset_name=dataset_name, input_len=input_len, output_len=output_len,
        use_timestamps=use_timestamps, use_clean_targets=USE_CLEAN_TARGETS,
        gpus=gpu_id, num_epochs=100, batch_size=64, callbacks=callbacks, seed=42,
        train_data_num_workers=16, val_data_num_workers=16, test_data_num_workers=16,
        train_data_pin_memory=True, val_data_pin_memory=True, test_data_pin_memory=True,
    )
    BasicTSLauncher.launch_training(cfg)


def worker_task(gpu_queue, model_name, dataset_name, num_features, input_len, output_len):
    """Worker process: Acquires GPU -> Runs Exp -> Releases GPU.
    
    Runs experiments based on EXPERIMENT_MODE:
        - "dropoutts": Only DropoutTS experiments
        - "sl": Only Selective Learning experiments
        - "dropoutts+sl": DropoutTS combined with SL
        - "all": All combinations (raw, dropoutts, sl, dropoutts+sl)
    """
    gpu_id = None
    task_id = f"{model_name} {dataset_name} ({input_len}->{output_len})"
    
    try:
        gpu_id = gpu_queue.get()
        print(f"[Start] {task_id} on GPU {gpu_id}")
        
        experiments = []
        
        # Build experiment list based on mode
        if EXPERIMENT_MODE in ["all"]:
            # 1. Raw baseline (no callbacks)
            experiments.append({
                "name": "Raw",
                "enable_dropout_ts": False,
                "enable_sl": False
            })
        
        if EXPERIMENT_MODE in ["dropoutts", "all"]:
            # 2. DropoutTS only experiments
            dt_combinations = list(product(
                HPARAMS["p_min"], HPARAMS["p_max"], HPARAMS["init_alpha"], 
                HPARAMS["init_sensitivity"], HPARAMS["sparsity_weight"], HPARAMS["use_gate"]
            ))
            for p_min, p_max, alpha, sens, sp_weight, use_gate in dt_combinations:
                if p_max <= p_min: continue
                experiments.append({
                    "name": f"DT(sens={sens},sp={sp_weight},gate={use_gate})",
                    "enable_dropout_ts": True, "enable_sl": False,
                    "p_min": p_min, "p_max": p_max, "init_alpha": alpha,
                    "init_sensitivity": sens, "sparsity_weight": sp_weight, "use_gate": use_gate
                })
        
        if EXPERIMENT_MODE in ["sl", "all"]:
            # 3. Selective Learning only experiments
            for r_u in SL_HPARAMS.get("r_u", [None]):
                for r_a in SL_HPARAMS.get("r_a", [None]):
                    if r_u is None and r_a is None:
                        continue  # Skip if both are None
                    exp = {
                        "name": f"SL(r_u={r_u},r_a={r_a})",
                        "enable_dropout_ts": False, "enable_sl": True,
                    }
                    if r_u is not None:
                        exp["r_u"] = r_u
                    if r_a is not None:
                        if ESTIMATOR_CKPT_PATH is None:
                            print(f"  [Warning] r_a={r_a} requires ESTIMATOR_CKPT_PATH, skipping...")
                            continue
                        exp["r_a"] = r_a
                        exp["ckpt_path"] = ESTIMATOR_CKPT_PATH
                    experiments.append(exp)
        
        if EXPERIMENT_MODE in ["dropoutts+sl", "all"]:
            # 4. DropoutTS + Selective Learning combined
            dt_combinations = list(product(
                HPARAMS["p_min"], HPARAMS["p_max"], HPARAMS["init_alpha"], 
                HPARAMS["init_sensitivity"], HPARAMS["sparsity_weight"], HPARAMS["use_gate"]
            ))
            for p_min, p_max, alpha, sens, sp_weight, use_gate in dt_combinations:
                if p_max <= p_min: continue
                for r_u in SL_HPARAMS.get("r_u", [None]):
                    for r_a in SL_HPARAMS.get("r_a", [None]):
                        if r_u is None and r_a is None:
                            continue  # Skip if both are None
                        exp = {
                            "name": f"DT+SL(sens={sens},gate={use_gate},r_u={r_u},r_a={r_a})",
                            "enable_dropout_ts": True, "enable_sl": True,
                            "p_min": p_min, "p_max": p_max, "init_alpha": alpha,
                            "init_sensitivity": sens, "sparsity_weight": sp_weight, "use_gate": use_gate,
                        }
                        if r_u is not None:
                            exp["r_u"] = r_u
                        if r_a is not None:
                            if ESTIMATOR_CKPT_PATH is None:
                                print(f"  [Warning] r_a={r_a} requires ESTIMATOR_CKPT_PATH, skipping...")
                                continue
                            exp["r_a"] = r_a
                            exp["ckpt_path"] = ESTIMATOR_CKPT_PATH
                        experiments.append(exp)
        
        total_exps = len(experiments)
        print(f"  -> Plan to run {total_exps} experiments (mode: {EXPERIMENT_MODE})")

        for idx, exp in enumerate(experiments):
            exp_name = exp.pop("name")
            print(f"    [Exp {idx+1}/{total_exps}] {exp_name}")
            
            run_experiment(
                model_name, dataset_name, num_features, input_len, output_len, str(gpu_id),
                **exp
            )

    except Exception as e:
        print(f"[Error] {task_id} failed: {e}")
        import traceback; traceback.print_exc()
    finally:
        if gpu_id is not None:
            gpu_queue.put(gpu_id)
            print(f"[Done] {task_id} released GPU {gpu_id}")


if __name__ == "__main__":
    # Initialize GPU Queue
    gpu_queue = Queue()
    for gpu_id in AVAILABLE_GPUS:
        gpu_queue.put(gpu_id)

    processes = []
    print(f"Scheduling tasks on GPUs: {AVAILABLE_GPUS}")

    # Schedule processes (One process per input/output length combination)
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

    print(f"Scheduled {len(processes)} tasks.")
    
    for p in processes:
        p.join()
    
    print("All experiments finished.")
