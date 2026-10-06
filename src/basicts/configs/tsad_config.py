from dataclasses import dataclass, field
from typing import Callable, List, Literal, Tuple, Union

import numpy as np
from basicts.data import BasicTSAnomalyDetectionDataset
from basicts.runners import BasicTSAnomalyDetectionRunner
from basicts.runners.callback import BasicTSCallback
from basicts.runners.taskflow import (BasicTSAnomalyDetectionTaskFlow,
                                      BasicTSTaskFlow)
from basicts.scaler import ZScoreScaler
from torch.optim import Adam

from .base_config import BasicTSConfig
from .model_config import BasicTSModelConfig


@dataclass(init=False)
class BasicTSAnomalyDetectionConfig(BasicTSConfig):
    """BasicTS config for reconstruction-based time series anomaly detection."""

    model: type = field(metadata={"help": "Model class. Must be specified."})
    model_config: BasicTSModelConfig = field(metadata={"help": "Model configuration. Must be specified."})
    dataset_name: str = field(default=None)

    gpus: Union[str, None] = None
    gpu_num: int = 0
    seed: int = 42
    taskflow: BasicTSTaskFlow = field(default=BasicTSAnomalyDetectionTaskFlow())
    runner_type: type = field(default=BasicTSAnomalyDetectionRunner)
    callbacks: List[BasicTSCallback] = field(default_factory=list)
    ddp_find_unused_parameters: bool = False
    compile_model: bool = False

    dataset_type: type = field(default=BasicTSAnomalyDetectionDataset)
    dataset_params: Union[dict, None] = None
    input_len: int = 96
    output_len: int = 96
    memmap: bool = False
    null_val: float = np.nan
    null_to_num: float = 0.0
    batch_size: Union[int, None] = None

    scaler: type = field(default=ZScoreScaler)
    norm_each_channel: bool = True
    rescale: bool = False

    metrics: List[Union[str, Tuple[str, Callable]]] = field(
        default_factory=lambda: [
            "AnomalyPrecision",
            "AnomalyRecall",
            "AnomalyF1",
            "AnomalyAdjustedPrecision",
            "AnomalyAdjustedRecall",
            "AnomalyAdjustedF1",
        ])
    target_metric: str = "loss"
    best_metric: Literal["min", "max"] = "min"

    num_epochs: int = 100
    num_steps: Union[int, None] = None
    loss: Union[str, Callable] = "MAE"
    optimizer: type = field(default=Adam)
    optimizer_params: dict = field(default_factory=lambda: {"lr": 2e-4, "weight_decay": 5e-4})
    lr: float = 2e-4
    lr_scheduler: Union[type, None] = None
    lr_scheduler_params: Union[dict, None] = None
    ckpt_save_dir: str = None
    ckpt_save_strategy: Union[int, List[int], Tuple[int]] = None
    finetune_from: Union[str, None] = None
    strict_load: bool = True

    train_batch_size: int = 64
    train_data_prefetch: bool = False
    train_data_shuffle: bool = True
    train_data_collate_fn: Union[Callable, None] = None
    train_data_num_workers: int = 0
    train_data_pin_memory: bool = False
    train_sample_ratio: float = 1.0

    val_batch_size: int = 64
    val_interval: int = 1
    val_data_prefetch: bool = False
    val_data_shuffle: bool = False
    val_data_collate_fn: Union[Callable, None] = None
    val_data_num_workers: int = 0
    val_data_pin_memory: bool = False

    test_batch_size: int = 64
    test_interval: int = 1
    test_data_prefetch: bool = False
    test_data_shuffle: bool = False
    test_data_collate_fn: Union[Callable, None] = None
    test_data_num_workers: int = 0
    test_data_pin_memory: bool = False

    anomaly_threshold: Union[float, None] = None
    anomaly_score_percentile: float = 95.0
    eval_after_train: bool = True
    save_results: bool = False

    tf32: bool = False
    deterministic: bool = False
    cudnn_enabled: bool = True
    cudnn_benchmark: bool = True
    cudnn_determinstic: bool = False

    def __post_init__(self):
        if self.ckpt_save_dir is None:
            self.ckpt_save_dir = (
                f"checkpoints/{self.model.__name__}/"
                f"{self.dataset_name}_anomaly_{self.num_epochs}_{self.input_len}"
            )
