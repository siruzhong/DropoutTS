import os
import shutil
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
from basicts.models.iTransformer import iTransformerForReconstruction, iTransformerConfig
from basicts.models.PatchTST import PatchTSTForReconstruction, PatchTSTConfig
from basicts.models.NonstationaryTransformer import (
    NonstationaryTransformerConfig,
    NonstationaryTransformerForReconstruction,
)
from basicts.models.AnomalyTransformer import (
    AnomalyTransformerConfig,
    AnomalyTransformerForReconstruction,
)
from basicts.models.DCdetector import DCdetectorConfig, DCdetectorForReconstruction
from basicts.models.TimesNet import TimesNetConfig, TimesNetForReconstruction
from basicts.models.TranAD import TranADConfig, TranADForReconstruction
from basicts.configs import BasicTSAnomalyDetectionConfig
from basicts.runners.callback import EarlyStopping, DropoutTSCallback
from basicts import BasicTSLauncher


# --- Global Configurations ---
AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
JOBS_PER_GPU = 4
MODELS = [
    "PatchTST",
    "iTransformer",
    "NonstationaryTransformer",
    "AnomalyTransformer",
    "TranAD",
    "DCdetector",
    "TimesNet",
]
DATASETS = [
    ("SMD", None),
    ("MSL", None),
    ("SMAP", None),
    ("PSM", None),
]

# Length settings
DATASET_CONFIGS = {
    "SMD": {"input_lens": [100]},
    "MSL": {"input_lens": [100]},
    "SMAP": {"input_lens": [100]},
    "PSM": {"input_lens": [100]},
    "default": {"input_lens": [100]},
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
    "anomaly_score_percentile": [95.0],
}
EXPERIMENT_SETTINGS = ["raw_baseline", "dropoutts"]

DATA_ROOTS = [os.path.join(script_dir, "datasets")]
RAW_DATA_DIRS = [os.path.join(script_dir, "datasets", "raw_data", "anomaly")]


def _load_array(path):
    """Load .npy/.npz/.csv/.txt files commonly used by TSAD datasets."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    if ext == ".npz":
        data = np.load(path)
        key = data.files[0]
        return data[key]
    if ext in [".csv", ".txt"]:
        try:
            return np.loadtxt(path, delimiter=",", skiprows=1)
        except ValueError:
            return np.loadtxt(path, delimiter=",")
    raise ValueError(f"Unsupported raw file type: {path}")


def _save_split(dataset_dir, mode, data, labels=None):
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    np.save(os.path.join(dataset_dir, f"{mode}_data.npy"), data)

    if labels is not None:
        labels = np.asarray(labels)
        if labels.ndim > 1:
            labels = labels.max(axis=tuple(range(1, labels.ndim)))
        np.save(os.path.join(dataset_dir, f"{mode}_labels.npy"), labels.astype(np.float32))


def _copy_existing_npy(raw_dir, dataset_dir):
    copied = False
    name_map = {
        "train_data.npy": "train_data.npy",
        "val_data.npy": "val_data.npy",
        "test_data.npy": "test_data.npy",
        "train_labels.npy": "train_labels.npy",
        "val_labels.npy": "val_labels.npy",
        "test_labels.npy": "test_labels.npy",
        "test_label.npy": "test_labels.npy",
        "labels.npy": "test_labels.npy",
    }
    for src_name, dst_name in name_map.items():
        src = os.path.join(raw_dir, src_name)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(dataset_dir, dst_name))
            copied = True
    return copied


def _find_first(raw_dir, candidates):
    for name in candidates:
        path = os.path.join(raw_dir, name)
        if os.path.exists(path):
            return path
    return None


def _load_indexed_csv_values(path):
    values = np.genfromtxt(path, delimiter=",", skip_header=1)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[1] > 1:
        values = values[:, 1:]
    return values


def _prepare_psm_from_raw(dataset_dir, raw_dir):
    train_path = os.path.join(raw_dir, "train.csv")
    test_path = os.path.join(raw_dir, "test.csv")
    label_path = os.path.join(raw_dir, "test_label.csv")
    if not all(os.path.exists(path) for path in [train_path, test_path, label_path]):
        return False

    train = np.nan_to_num(_load_indexed_csv_values(train_path), nan=0.0, posinf=0.0, neginf=0.0)
    test = np.nan_to_num(_load_indexed_csv_values(test_path), nan=0.0, posinf=0.0, neginf=0.0)
    labels = _load_indexed_csv_values(label_path)
    val_size = max(1, int(len(train) * 0.2))

    _save_split(dataset_dir, "train", train[:-val_size])
    _save_split(dataset_dir, "val", train[-val_size:])
    _save_split(dataset_dir, "test", test, labels)
    print(f"[Data] Prepared PSM from {raw_dir} with NaNs filled by 0.0")
    return True


def _processed_data_has_nan(dataset_dir):
    for name in ["train_data.npy", "val_data.npy", "test_data.npy"]:
        path = os.path.join(dataset_dir, name)
        if not os.path.exists(path):
            return False
        data = np.load(path, mmap_mode="r")
        if np.isnan(data).any():
            return True
    return False


def _prepare_from_common_raw(dataset_name, dataset_dir, raw_dir):
    train_path = _find_first(raw_dir, [
        "train.npy", "train.csv", "train.txt", "SMD_train.npy", "SMD_train.csv"
    ])
    test_path = _find_first(raw_dir, [
        "test.npy", "test.csv", "test.txt", "SMD_test.npy", "SMD_test.csv"
    ])
    label_path = _find_first(raw_dir, [
        "test_label.npy", "test_label.csv", "test_labels.npy", "test_labels.csv",
        "labels.npy", "labels.csv", "test_label.txt"
    ])

    if train_path is None or test_path is None:
        return False

    train = _load_array(train_path)
    test = _load_array(test_path)
    labels = _load_array(label_path) if label_path is not None else None
    val_size = max(1, int(len(train) * 0.2))

    _save_split(dataset_dir, "train", train[:-val_size])
    _save_split(dataset_dir, "val", train[-val_size:])
    _save_split(dataset_dir, "test", test, labels)
    print(f"[Data] Prepared {dataset_name} from {raw_dir}")
    return True


def _dataset_dir_candidates(dataset_name):
    for root in DATA_ROOTS:
        yield os.path.join(root, dataset_name)


def prepare_dataset(dataset_name):
    """Prepare BasicTS anomaly files when local raw files are available."""
    required = ["train_data.npy", "val_data.npy", "test_data.npy"]
    raw_candidates = [
        os.path.join(raw_root, dataset_name)
        for raw_root in RAW_DATA_DIRS
    ]
    for dataset_dir in _dataset_dir_candidates(dataset_name):
        if all(os.path.exists(os.path.join(dataset_dir, name)) for name in required):
            if dataset_name == "PSM" and _processed_data_has_nan(dataset_dir):
                for raw_dir in raw_candidates:
                    if os.path.isdir(raw_dir) and _prepare_psm_from_raw(dataset_dir, raw_dir):
                        return dataset_dir
            return dataset_dir

    dataset_dir = os.path.join(DATA_ROOTS[-1], dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    raw_candidates = raw_candidates + [
        os.path.join(dataset_dir, "raw"),
    ]

    for raw_dir in raw_candidates:
        if not os.path.isdir(raw_dir):
            continue
        if dataset_name == "PSM" and _prepare_psm_from_raw(dataset_dir, raw_dir):
            return dataset_dir
        if _copy_existing_npy(raw_dir, dataset_dir):
            if all(os.path.exists(os.path.join(dataset_dir, name)) for name in required):
                print(f"[Data] Copied prepared {dataset_name} npy files from {raw_dir}")
                return dataset_dir
        if _prepare_from_common_raw(dataset_name, dataset_dir, raw_dir):
            return dataset_dir

    raise FileNotFoundError(
        f"Dataset {dataset_name} is not prepared. Put processed files under datasets/{dataset_name}/ "
        f"or raw files under datasets/raw_data/anomaly/{dataset_name}/. Expected train/val/test_data.npy "
        f"and optional *_labels.npy."
    )


def get_num_features(dataset_name):
    dataset_dir = prepare_dataset(dataset_name)
    data = np.load(os.path.join(dataset_dir, "train_data.npy"), mmap_mode="r")
    return 1 if data.ndim == 1 else data.shape[-1]


def get_dataset_dir(dataset_name):
    return prepare_dataset(dataset_name)


def get_model_config(model_name, input_len, num_features):
    """Factory to create reconstruction model class and config."""
    if model_name == "PatchTST":
        cfg = PatchTSTConfig(
            input_len=input_len,
            output_len=input_len,
            num_features=num_features,
        )
        return PatchTSTForReconstruction, cfg

    elif model_name == "iTransformer":
        cfg = iTransformerConfig(
            input_len=input_len,
            output_len=input_len,
            num_features=num_features,
        )
        return iTransformerForReconstruction, cfg

    elif model_name == "NonstationaryTransformer":
        cfg = NonstationaryTransformerConfig(
            input_len=input_len,
            output_len=input_len,
            num_features=num_features,
        )
        return NonstationaryTransformerForReconstruction, cfg

    elif model_name == "AnomalyTransformer":
        cfg = AnomalyTransformerConfig(
            input_len=input_len,
            output_len=input_len,
            num_features=num_features,
        )
        return AnomalyTransformerForReconstruction, cfg

    elif model_name == "TranAD":
        cfg = TranADConfig(
            input_len=input_len,
            output_len=input_len,
            num_features=num_features,
        )
        return TranADForReconstruction, cfg

    elif model_name == "DCdetector":
        cfg = DCdetectorConfig(
            input_len=input_len,
            output_len=input_len,
            num_features=num_features,
        )
        return DCdetectorForReconstruction, cfg

    elif model_name == "TimesNet":
        cfg = TimesNetConfig(
            input_len=input_len,
            output_len=input_len,
            num_features=num_features,
            hidden_size=128,
            intermediate_size=256,
            num_layers=2,
        )
        return TimesNetForReconstruction, cfg

    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_experiment(model_name, dataset_name, num_features, input_len, gpu_id, **kwargs):
    """Setup and launch a single anomaly detection experiment."""
    model_class, model_config = get_model_config(model_name, input_len, num_features)

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

    dataset_dir = get_dataset_dir(dataset_name)
    cfg = BasicTSAnomalyDetectionConfig(
        model=model_class, model_config=model_config,
        dataset_name=dataset_name, input_len=input_len, output_len=input_len,
        dataset_params={"data_file_path": dataset_dir},
        train_sample_ratio=kwargs.get('train_sample_ratio', 1.0),
        anomaly_score_percentile=kwargs.get('anomaly_score_percentile', 95.0),
        gpus=gpu_id, num_epochs=100, batch_size=64, callbacks=callbacks, seed=42,
        train_data_num_workers=16, val_data_num_workers=16, test_data_num_workers=16,
        train_data_pin_memory=True, val_data_pin_memory=True, test_data_pin_memory=True,
    )
    metrics_path = os.path.join(cfg.ckpt_save_dir, cfg.md5, "test_metrics.json")
    if os.path.exists(metrics_path):
        print(f"    [Skip] Existing metrics at {metrics_path}")
        return
    BasicTSLauncher.launch_training(cfg)


def worker_task(gpu_queue, model_name, dataset_name, input_len, exp_setting):
    """Worker process: Acquires GPU -> Runs Exp -> Releases GPU."""
    gpu_id = None
    task_id = f"{model_name}/{dataset_name} ({input_len}) [{exp_setting}]"

    try:
        gpu_id = gpu_queue.get()
        print(f"[Start] {task_id} on GPU {gpu_id}")
        num_features = get_num_features(dataset_name)

        if exp_setting == "raw_baseline":
            print("  -> Running raw baseline without DropoutTS.")
            run_experiment(
                model_name, dataset_name, num_features, input_len, str(gpu_id),
                enable_dropout_ts=False,
                train_sample_ratio=1.0,
                anomaly_score_percentile=95.0,
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
            HPARAMS["train_sample_ratio"],
            HPARAMS["anomaly_score_percentile"],
        ))

        total_exps = len(param_combinations)
        print(f"  -> Plan to run {total_exps} experiments on this GPU.")

        for idx, (p_min, p_max, alpha, sens, sp_weight, use_gate, train_ratio, percentile) in enumerate(param_combinations):
            if p_max <= p_min:
                continue

            print(
                f"    [Exp {idx+1}/{total_exps}] sens={sens}, sp_weight={sp_weight}, "
                f"use_gate={use_gate}, train_ratio={train_ratio}, percentile={percentile}"
            )

            run_experiment(
                model_name, dataset_name, num_features, input_len, str(gpu_id),
                enable_dropout_ts=True,
                p_min=p_min, p_max=p_max,
                init_alpha=alpha, init_sensitivity=sens,
                sparsity_weight=sp_weight, use_gate=use_gate,
                train_sample_ratio=train_ratio,
                anomaly_score_percentile=percentile,
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
        for _ in range(JOBS_PER_GPU):
            gpu_queue.put(gpu_id)

    processes = []
    max_concurrent = len(AVAILABLE_GPUS) * JOBS_PER_GPU
    print(
        f"Scheduling anomaly detection tasks on GPUs: {AVAILABLE_GPUS} "
        f"(up to {max_concurrent} concurrent, JOBS_PER_GPU={JOBS_PER_GPU})"
    )

    # Schedule processes (One process per dataset/model/input length combination)
    for model_name in MODELS:
        for dataset_name, _ in DATASETS:
            config = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS["default"])

            for input_len in config["input_lens"]:
                for exp_setting in EXPERIMENT_SETTINGS:
                    p = Process(
                        target=worker_task,
                        args=(gpu_queue, model_name, dataset_name, input_len, exp_setting)
                    )
                    p.start()
                    processes.append(p)
                    time.sleep(0.1)

    print(f"Scheduled {len(processes)} tasks.")

    for p in processes:
        p.join()

    print("All anomaly detection experiments finished.")
