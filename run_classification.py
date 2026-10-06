import os
import sys
import time
from itertools import product
from multiprocessing import Process, Queue

import numpy as np

# Add src directory
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import Models
from basicts.models.iTransformer import iTransformerForClassification, iTransformerConfig
from basicts.models.PatchTST import PatchTSTConfig, PatchTSTForClassification
from basicts.models.NonstationaryTransformer import (
    NonstationaryTransformerConfig,
    NonstationaryTransformerForClassification,
)
from basicts.models.InceptionTime import (
    InceptionTimeConfig,
    InceptionTimeForClassification,
)
from basicts.models.MiniROCKET import MiniROCKETConfig, MiniROCKETForClassification
from basicts.models.TimesNet import TimesNetConfig, TimesNetForClassification
from basicts.configs import BasicTSClassificationConfig
from basicts.runners.callback import EarlyStopping, DropoutTSCallback
from basicts import BasicTSLauncher


# --- Global Configurations ---
AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
JOBS_PER_GPU = 1
VALIDATION_RATIO = 0.2
VALIDATION_SPLIT_SEED = 42
MODELS = [
    "iTransformer",
    "PatchTST",
    "NonstationaryTransformer",
    "TimesNet",
    "InceptionTime",
    "MiniROCKET",
]
DATASETS = [
    ("ArticularyWordRecognition", None),
    ("AtrialFibrillation", None),
    ("BasicMotions", None),
    ("CharacterTrajectories", None),
    ("Cricket", None),
    ("DuckDuckGeese", None),
    ("EigenWorms", None),
    ("Epilepsy", None),
    ("EthanolConcentration", None),
    ("ERing", None),
    ("FaceDetection", None),
    ("FingerMovements", None),
    ("HandMovementDirection", None),
    ("Handwriting", None),
    ("Heartbeat", None),
    ("InsectWingbeat", None),
    ("JapaneseVowels", None),
    ("Libras", None),
    ("LSST", None),
    ("MotorImagery", None),
    ("NATOPS", None),
    ("PenDigits", None),
    ("PEMS-SF", None),
    ("PhonemeSpectra", None),
    ("RacketSports", None),
    ("SelfRegulationSCP1", None),
    ("SelfRegulationSCP2", None),
    ("SpokenArabicDigits", None),
    ("StandWalkJump", None),
    ("UWaveGestureLibrary", None),
    ("HAR", None),
    ("SleepEDF", None),
]

DATA_ROOTS = [
    os.path.join(script_dir, "datasets", "UEA"),
]

# Hyperparameters
HPARAMS = {
    "p_min": [0.05],
    "p_max": [0.5],
    "init_alpha": [10.0],
    "init_sensitivity": [1.0],
    "sparsity_weight": [0.0],
    "use_gate": [False],
}
EXPERIMENT_SETTINGS = ["raw_baseline", "dropoutts"]

BATCH_SIZE_OVERRIDES = {
    ("PatchTST", "EigenWorms"): 1,
    ("PatchTST", "MotorImagery"): 1,
    ("NonstationaryTransformer", "EigenWorms"): 1,
    ("NonstationaryTransformer", "EthanolConcentration"): 1,
}


def get_batch_size(model_name, dataset_name):
    return BATCH_SIZE_OVERRIDES.get((model_name, dataset_name), 64)


def get_num_workers(model_name, dataset_name):
    if (model_name, dataset_name) in BATCH_SIZE_OVERRIDES:
        return 2
    return 16


def get_dataset_dir(dataset_name):
    required = ["train_inputs.npy", "train_labels.npy", "test_inputs.npy", "test_labels.npy"]
    for root in DATA_ROOTS:
        dataset_dir = os.path.join(root, dataset_name)
        if all(os.path.exists(os.path.join(dataset_dir, name)) for name in required):
            return dataset_dir
    raise FileNotFoundError(
        f"Dataset {dataset_name} is not prepared. Expected {required} under "
        f"datasets/UEA/{dataset_name}/."
    )


def get_dataset_meta(dataset_name):
    dataset_dir = get_dataset_dir(dataset_name)
    inputs = np.load(os.path.join(dataset_dir, "train_inputs.npy"), mmap_mode="r")
    train_labels = np.load(os.path.join(dataset_dir, "train_labels.npy"), mmap_mode="r")
    if inputs.ndim == 2:
        input_len, num_features = inputs.shape[1], 1
    else:
        input_len, num_features = inputs.shape[1], inputs.shape[2]
    num_classes = int(np.max(train_labels)) + 1
    return dataset_dir, input_len, num_features, num_classes


def get_model_config(model_name, input_len, num_features, num_classes):
    """Factory to create classification model class and config."""
    if model_name == "iTransformer":
        cfg = iTransformerConfig(
            input_len=input_len,
            num_features=num_features,
            num_classes=num_classes,
        )
        return iTransformerForClassification, cfg
    if model_name == "PatchTST":
        patch_len = 64 if input_len >= 1024 else 16
        patch_stride = 32 if input_len >= 1024 else 8
        cfg = PatchTSTConfig(
            input_len=input_len,
            num_features=num_features,
            num_classes=num_classes,
            patch_len=patch_len,
            patch_stride=patch_stride,
            hidden_size=64,
            intermediate_size=256,
        )
        return PatchTSTForClassification, cfg
    if model_name == "NonstationaryTransformer":
        if input_len >= 1024:
            cfg = NonstationaryTransformerConfig(
                input_len=input_len,
                num_features=num_features,
                num_classes=num_classes,
                hidden_size=64,
                proj_hidden_size=64,
                n_heads=1,
                intermediate_size=256,
                num_encoder_layers=1,
            )
            return NonstationaryTransformerForClassification, cfg
        cfg = NonstationaryTransformerConfig(
            input_len=input_len,
            num_features=num_features,
            num_classes=num_classes,
        )
        return NonstationaryTransformerForClassification, cfg
    if model_name == "TimesNet":
        cfg = TimesNetConfig(
            input_len=input_len,
            output_len=0,
            num_features=num_features,
            num_classes=num_classes,
            hidden_size=128,
            intermediate_size=256,
            num_layers=2,
        )
        return TimesNetForClassification, cfg
    if model_name == "InceptionTime":
        cfg = InceptionTimeConfig(
            input_len=input_len,
            num_features=num_features,
            num_classes=num_classes,
        )
        return InceptionTimeForClassification, cfg
    if model_name == "MiniROCKET":
        cfg = MiniROCKETConfig(
            input_len=input_len,
            num_features=num_features,
            num_classes=num_classes,
        )
        return MiniROCKETForClassification, cfg
    raise ValueError(f"Unknown model: {model_name}")


def run_experiment(model_name, dataset_name, gpu_id, **kwargs):
    """Setup and launch a single classification experiment."""
    dataset_dir, input_len, num_features, num_classes = get_dataset_meta(dataset_name)
    model_class, model_config = get_model_config(
        model_name, input_len, num_features, num_classes
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

    num_workers = get_num_workers(model_name, dataset_name)
    cfg = BasicTSClassificationConfig(
        model=model_class, model_config=model_config,
        dataset_name=dataset_name,
        dataset_params={
            "data_file_path": dataset_dir,
            "validation_ratio": VALIDATION_RATIO,
            "split_seed": VALIDATION_SPLIT_SEED,
        },
        gpus=gpu_id, num_epochs=100,
        batch_size=get_batch_size(model_name, dataset_name),
        callbacks=callbacks, seed=42,
        model_dtype="bfloat16",
        train_data_num_workers=num_workers,
        val_data_num_workers=num_workers,
        test_data_num_workers=num_workers,
        train_data_pin_memory=False,
        val_data_pin_memory=False,
        test_data_pin_memory=False,
    )
    metrics_path = os.path.join(cfg.ckpt_save_dir, cfg.md5, "test_metrics.json")
    if os.path.exists(metrics_path):
        print(f"    [Skip] Existing metrics at {metrics_path}")
        return
    BasicTSLauncher.launch_training(cfg)


def worker_task(gpu_queue, model_name, dataset_name, exp_setting):
    """Worker process: Acquires GPU -> Runs Exp -> Releases GPU."""
    gpu_id = None
    task_id = f"{model_name}/{dataset_name} [{exp_setting}]"

    try:
        gpu_id = gpu_queue.get()
        print(f"[Start] {task_id} on GPU {gpu_id}")

        if exp_setting == "raw_baseline":
            print("  -> Running raw baseline without DropoutTS.")
            run_experiment(
                model_name, dataset_name, str(gpu_id),
                enable_dropout_ts=False,
            )
            return

        if exp_setting != "dropoutts":
            raise ValueError(f"Unknown experiment setting: {exp_setting}")

        param_combinations = list(product(
            HPARAMS["p_min"],
            HPARAMS["p_max"],
            HPARAMS["init_alpha"],
            HPARAMS["init_sensitivity"],
            HPARAMS["sparsity_weight"],
            HPARAMS["use_gate"],
        ))

        total_exps = len(param_combinations)
        print(f"  -> Plan to run {total_exps} experiments on this GPU.")

        for idx, (p_min, p_max, alpha, sens, sp_weight, use_gate) in enumerate(param_combinations):
            if p_max <= p_min:
                continue

            print(f"    [Exp {idx+1}/{total_exps}] sens={sens}, sp_weight={sp_weight}, use_gate={use_gate}")

            run_experiment(
                model_name, dataset_name, str(gpu_id),
                enable_dropout_ts=True,
                p_min=p_min, p_max=p_max,
                init_alpha=alpha, init_sensitivity=sens,
                sparsity_weight=sp_weight, use_gate=use_gate,
            )

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
        for dataset_name, _ in DATASETS:
            for exp_setting in EXPERIMENT_SETTINGS:
                p = Process(
                    target=worker_task,
                    args=(gpu_queue, model_name, dataset_name, exp_setting)
                )
                p.start()
                processes.append(p)
                time.sleep(0.1)

    print(f"Scheduled {len(processes)} tasks.")

    for p in processes:
        p.join()

    print("All experiments finished.")
