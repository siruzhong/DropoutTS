"""
Pretrain DLinear as estimator for Selective Learning.

This script trains a DLinear model on the specified dataset,
which can then be used as the estimator for anomaly mask in SelectiveLearning callback.

Usage:
    python scripts/run_pretrain_estimator.py
"""

import sys
import os
import time
from multiprocessing import Process, Queue

# Add src directory (the repository root is one level above this script)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(project_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from basicts.models.DLinear import DLinear, DLinearConfig
from basicts.configs import BasicTSForecastingConfig
from basicts.runners.callback import EarlyStopping
from basicts import BasicTSLauncher


# --- Configuration ---
AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
JOBS_PER_GPU = 4
SEED = 42
PRETRAIN_TASKS = [
    {
        "dataset_name": "Illness",
        "num_features": 7,
        "input_len": 24,
        "output_len": 24,
    },
]


def run_experiment(dataset_name, num_features, input_len, output_len, gpu_id):
    """Train DLinear estimator."""

    # Create DLinear config
    model_config = DLinearConfig(
        input_len=input_len,
        output_len=output_len,
        num_features=num_features,
        moving_avg=25,
        individual=False
    )
    
    # Create training config
    cfg = BasicTSForecastingConfig(
        model=DLinear,
        model_config=model_config,
        dataset_name=dataset_name,
        input_len=input_len,
        output_len=output_len,
        use_timestamps=False,
        use_clean_targets=True,  # Use clean targets for estimator training
        gpus=gpu_id,
        num_epochs=100,
        batch_size=64,
        callbacks=[EarlyStopping(patience=10)],
        seed=SEED,
        train_data_num_workers=8,
        val_data_num_workers=8,
        test_data_num_workers=8,
        train_data_pin_memory=True,
        val_data_pin_memory=True,
        test_data_pin_memory=True,
    )
    
    print("=" * 60)
    print("Pretraining DLinear Estimator for Selective Learning")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Input Length: {input_len}")
    print(f"Output Length: {output_len}")
    print(f"GPU: {gpu_id}")
    print("=" * 60)

    # Launch training
    BasicTSLauncher.launch_training(cfg)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nTo use this estimator in SelectiveLearning, find the checkpoint at:")
    print(f"  checkpoints/DLinear/{dataset_name}_*_{input_len}_{output_len}/*/best_model.pt")
    print("\nThen configure run_baselines.py like this:")
    print("""
# In run_baselines.py, update run_experiment() to pass estimator:

from basicts.models.DLinear import DLinear, DLinearConfig

# Create estimator instance
estimator_config = DLinearConfig(
    input_len=96,
    output_len=720, 
    num_features=1
)
estimator = DLinear(estimator_config)

# Pass to SelectiveLearning callback
SelectiveLearning(
    r_a=0.2,
    estimator=estimator,
    ckpt_path="checkpoints/DLinear/.../best_model.pt"
)
""")


def worker_task(gpu_queue, task):
    gpu_id = None
    task_id = f"{task['dataset_name']} ({task['input_len']}->{task['output_len']})"
    try:
        gpu_id = gpu_queue.get()
        print(f"[Start] {task_id} on GPU {gpu_id}")
        run_experiment(
            task["dataset_name"],
            task["num_features"],
            task["input_len"],
            task["output_len"],
            str(gpu_id),
        )
    except Exception as e:
        print(f"[Error] {task_id} failed: {e}")
        import traceback; traceback.print_exc()
    finally:
        if gpu_id is not None:
            gpu_queue.put(gpu_id)
            print(f"[Done] {task_id} released GPU {gpu_id}")


def main():
    gpu_queue = Queue()
    for gpu_id in AVAILABLE_GPUS:
        for _ in range(JOBS_PER_GPU):
            gpu_queue.put(gpu_id)

    processes = []
    max_concurrent = len(AVAILABLE_GPUS) * JOBS_PER_GPU
    print(
        f"Scheduling estimator pretraining on GPUs: {AVAILABLE_GPUS} "
        f"(up to {max_concurrent} concurrent, JOBS_PER_GPU={JOBS_PER_GPU})"
    )

    for task in PRETRAIN_TASKS:
        p = Process(target=worker_task, args=(gpu_queue, task))
        p.start()
        processes.append(p)
        time.sleep(0.1)

    print(f"Scheduled {len(processes)} estimator pretraining tasks.")

    for p in processes:
        p.join()

    print("All estimator pretraining tasks finished.")


if __name__ == "__main__":
    main()
