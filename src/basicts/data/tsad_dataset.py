import os
from typing import Union

import numpy as np
from basicts.utils import BasicTSMode

from .base_dataset import BasicTSDataset


class BasicTSAnomalyDetectionDataset(BasicTSDataset):
    """Windowed dataset for reconstruction-based time series anomaly detection."""

    def __init__(
            self,
            dataset_name: str,
            input_len: int,
            mode: Union[BasicTSMode, str],
            local: bool = True,
            data_file_path: Union[str, None] = None,
            memmap: bool = False) -> None:
        super().__init__(dataset_name, mode, memmap)
        self.input_len = input_len
        if not local:
            pass  # TODO: support remote anomaly datasets.
        if data_file_path is None:
            data_file_path = f"datasets/{dataset_name}"

        try:
            self._data = np.load(
                os.path.join(data_file_path, f"{mode}_data.npy"),
                mmap_mode="r" if memmap else None)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Cannot load anomaly dataset from {data_file_path}. "
                f"Expected {mode}_data.npy."
            ) from e

        label_path = os.path.join(data_file_path, f"{mode}_labels.npy")
        if os.path.exists(label_path):
            self._labels = np.load(label_path, mmap_mode="r" if memmap else None)
        else:
            self._labels = np.zeros((self._data.shape[0],), dtype=np.float32)

        if self._data.ndim == 1:
            self._data = self._data[:, None]
        if self._labels.ndim > 1:
            self._labels = np.max(self._labels, axis=tuple(range(1, self._labels.ndim)))
        self.memmap = memmap

    def __getitem__(self, index: int) -> dict:
        inputs = self._data[index: index + self.input_len]
        labels = self._labels[index: index + self.input_len]
        return {
            "inputs": inputs.copy() if self.memmap else inputs,
            "targets": inputs.copy() if self.memmap else inputs,
            "labels": labels.copy() if self.memmap else labels,
            "index": index,
        }

    def __len__(self) -> int:
        return max(0, len(self._data) - self.input_len + 1)

    @property
    def data(self) -> np.ndarray:
        return self._data
