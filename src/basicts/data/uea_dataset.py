import os
from typing import Union

import numpy as np
from basicts.utils import BasicTSMode

from .base_dataset import BasicTSDataset


class UEADataset(BasicTSDataset):
    """
    A UEA dataset class for time series classification problems, handling the loading, parsing, and partitioning
    of time series data into training, validation, and testing sets based on provided ratios.
    
    This class supports configurations where sequences may or may not overlap, accommodating scenarios
    where time series data is drawn from continuous periods or distinct episodes, affecting how
    the data is split into batches for model training or evaluation.
    
    Attributes:
        data_file_path (str): Path to the file containing the time series data.
        description_file_path (str): Path to the JSON file containing the description of the dataset.
        data (np.ndarray): The loaded time series data array, split according to the specified mode.
        description (dict): Metadata about the dataset, such as shape and other properties.
    """

    def __init__(
            self,
            dataset_name: str,
            mode: Union[BasicTSMode, str],
            local: bool = True,
            data_file_path: Union[str, None] = None,
            memmap: bool = False,
            validation_ratio: float = 0.2,
            split_seed: int = 42) -> None:
        """
        Initializes the UEADataset by setting up paths, loading data, and 
        preparing it according to the specified configurations.

        Args:
             dataset_name (str): The name of the dataset.
             mode (BasicTSMode | str): The operation mode of the dataset. Valid values are "train", "val", or "test".
             local (bool, optional): Flag to determine if the dataset should be loaded locally. Defaults to True.
             data_file_path (str | None, optional): Path to the file containing the time series data. Defaults to None.
             memmap (bool, optional): Flag to determine if the dataset should be loaded using memory mapping. Defaults to False.
             validation_ratio (float): Fraction held out within each training class when validation files are absent.
             split_seed (int): Seed for the shared train/validation partition.
        """

        mode = BasicTSMode(mode)
        super().__init__(dataset_name, mode, memmap)
        if not local:
            pass # TODO: support download remotely
        self.data_file_path = data_file_path or f"datasets/UEA/{dataset_name}" # default file path
        validation_files = [os.path.join(self.data_file_path, f"val_{name}.npy")
                            for name in ("inputs", "labels")]
        validation_exists = [os.path.isfile(path) for path in validation_files]
        needs_split = mode in (BasicTSMode.TRAIN, BasicTSMode.VAL) and not all(validation_exists)
        if needs_split and any(validation_exists):
            raise FileNotFoundError("Provide both val_inputs.npy and val_labels.npy, or neither.")
        source_mode = BasicTSMode.TRAIN if needs_split else mode
        try:
            self.inputs = np.load(
                os.path.join(self.data_file_path, f"{source_mode}_inputs.npy"),
                mmap_mode="r" if memmap else None)
            self.labels = np.load(
                os.path.join(self.data_file_path, f"{source_mode}_labels.npy"),
                mmap_mode="r" if memmap else None)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Cannot load dataset from {self.data_file_path}, Please set a correct local path."\
                                    "If you want to download the dataset, please set the argument `local` to False.") from e
        self.memmap = memmap

        # to be compatible with the original dataset shape
        if self.inputs.ndim == 4:
            self.inputs = np.squeeze(self.inputs, axis=-1)

        self._indices = None
        if needs_split:
            if not 0 < validation_ratio < 1:
                raise ValueError("validation_ratio must be between 0 and 1.")
            if self.labels.ndim != 1 or len(self.labels) != len(self.inputs):
                raise ValueError("Expected one classification label per training sample.")
            rng = np.random.default_rng(split_seed)
            validation_mask = np.zeros(len(self.labels), dtype=bool)
            for label in np.unique(self.labels):
                indices = np.flatnonzero(self.labels == label)
                if len(indices) < 2:
                    continue  # Keep singleton classes in training.
                rng.shuffle(indices)
                count = min(len(indices) - 1, max(1, round(len(indices) * validation_ratio)))
                validation_mask[indices[:count]] = True
            if not validation_mask.any():
                raise ValueError("Cannot form a validation holdout: provide independent val_inputs.npy and val_labels.npy.")
            self._indices = np.flatnonzero(validation_mask if mode == BasicTSMode.VAL else ~validation_mask)

    def __getitem__(self, index: int) -> dict:
        """
        Retrieves a sample from the dataset at the specified index, considering both the input and output lengths.

        Args:
            index (int): The index of the desired sample in the dataset.

        Returns:
            dict: A dictionary containing "inputs" and "target", where both are slices of the dataset corresponding to
                  the historical input data and future prediction data, respectively.
        """

        if self._indices is not None:
            index = int(self._indices[index])
        if self.memmap:
            return {
                "inputs": self.inputs[index,...].copy(),
                "targets": self.labels[index].copy()
                }

        else:
            return {
                "inputs": self.inputs[index,...],
                "targets": self.labels[index]
                }

    def __len__(self) -> int:
        """
        Calculates the total number of samples available in the dataset, adjusted for the lengths of input and output sequences.

        Returns:
            int: The number of valid samples that can be drawn from the dataset, based on the configurations of input and output lengths.
        """
        return len(self._indices) if self._indices is not None else self.inputs.shape[0]

    @property
    def data(self) -> np.ndarray:
        return self.inputs[self._indices] if self._indices is not None else self.inputs
