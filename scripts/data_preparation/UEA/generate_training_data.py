import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sktime.datasets import load_from_tsfile_to_dataframe


@dataclass
class UEAProcessor:

    '''
    Process UEA datasets.
    '''

    num_classes: int = None
    class_names: list = None
    equal_length: bool = None
    missing: bool = None
    filling_missing_value: str = 'linear'
    norm_each_channel: bool = True
    mean: np.ndarray = None
    stats: dict = field(default_factory=dict)
    validation_ratio: float = 0.2
    split_seed: int = 42

    def run(self):
        for item in os.listdir('datasets/raw_data/UEA'):
            train_file_path = os.path.join('datasets/raw_data/UEA', item, f'{item}_TRAIN.ts')
            train_x, train_y = self.load_data(train_file_path)
            train_indices, val_indices = self.split_training_indices(train_y)
            val_x, val_y = train_x.iloc[val_indices], np.asarray(train_y)[val_indices]
            train_x, train_y = train_x.iloc[train_indices], np.asarray(train_y)[train_indices]
            max_len = int(train_x.map(len).values.max())
            train_x, train_y = self.preprocess_data(train_x, train_y, max_len, True)
            val_x, val_y = self.preprocess_data(val_x, val_y, max_len, False)
            test_file_path = os.path.join('datasets/raw_data/UEA', item, f'{item}_TEST.ts')
            test_x, test_y = self.load_data(test_file_path)
            test_x, test_y = self.preprocess_data(test_x, test_y, max_len, False)

            output_path = os.path.join('datasets/UEA', item)
            os.makedirs(output_path, exist_ok=True)

            self.save_data(output_path, train_x, train_y, test_x, test_y)
            np.save(os.path.join(output_path, 'val_inputs.npy'), val_x)
            np.save(os.path.join(output_path, 'val_labels.npy'), val_y)

            description = {
                'name': item,
                'num_classes': self.num_classes,
                'class_names': self.class_names,
                'equal_length': self.equal_length,
                'seq_len': train_x.shape[1],
                'num_nodes': train_x.shape[2],
                'num_features': train_x.shape[3],
                'shape': '[num_samples, seq_len, num_nodes, num_features]',
                'missing': self.missing,
                'filling_missing': self.filling_missing_value if self.missing else 'NA',
                'norm_each_channel': self.norm_each_channel,
                'validation_ratio': self.validation_ratio,
                'split_seed': self.split_seed
            }
            description_path = os.path.join(output_path, 'desc.json')
            with open(description_path, 'w') as f:
                json.dump(description, f, indent=4)
            print(f'{item} is finished.')
            self.reset()

    def split_training_indices(self, labels):
        if not 0 < self.validation_ratio < 1:
            raise ValueError('validation_ratio must be between 0 and 1.')
        labels = np.asarray(labels)
        rng = np.random.default_rng(self.split_seed)
        validation_mask = np.zeros(len(labels), dtype=bool)
        for label in np.unique(labels):
            indices = np.flatnonzero(labels == label)
            if len(indices) < 2:
                continue
            rng.shuffle(indices)
            count = min(len(indices) - 1, max(1, round(len(indices) * self.validation_ratio)))
            validation_mask[indices[:count]] = True
        if not validation_mask.any():
            raise ValueError('Cannot form a validation holdout from the training samples.')
        return np.flatnonzero(~validation_mask), np.flatnonzero(validation_mask)

    def reset(self):
        self.num_classes = None
        self.class_names = None
        self.equal_length = None
        self.missing = None
        self.mean = None
        self.stats = {}

    def load_data(self, file_path: str):
        '''Load and preprocess raw data, selecting the specified channel(s).'''

        inputs, labels = load_from_tsfile_to_dataframe(file_path,
                                                    return_separate_X_and_y=True,
                                                    replace_missing_vals_with='NaN')
        return inputs, labels

    def preprocess_data(self, inputs, labels, seq_len, is_train):

        # Convert labels to integer
        if is_train:
            self.class_names = pd.Series(labels, dtype='category').cat.categories.to_list()
            self.num_classes = len(self.class_names)
        labels = np.asarray(pd.Categorical(labels, categories=self.class_names).codes, dtype=np.int64)
        if np.any(labels < 0):
            raise ValueError('Validation/test labels must belong to the training label vocabulary.')

        # Fill missing values
        if self.missing is None or not self.missing:
            has_nan = inputs.map(
                lambda x: x.isna().any()
            ).values
            self.missing = bool(has_nan.any())
        if self.missing:
            inputs = inputs.map(
                lambda x: x.interpolate(method=self.filling_missing_value, limit_direction='both'))

        x = inputs.map(lambda x: self.align_series(x, seq_len))
        x = np.array(x.values.tolist(), dtype=np.float32)

        if self.equal_length is None or self.equal_length:
            self.equal_length = not np.isnan(x).any()

        if is_train:
            if self.norm_each_channel:
                self.stats['mean'] = np.nanmean(x, axis=(0,2), keepdims=True)
                self.stats['std'] = np.nanstd(x, axis=(0,2), keepdims=True)
                self.stats['std'][self.stats['std'] == 0] = 1.0  # prevent division by zero by setting std to 1 where it's 0
            else:
                self.stats['mean'] = np.nanmean(x)
                self.stats['std'] = np.nanstd(x)
                if self.stats['std'] == 0:
                    self.stats['std'] = 1.0  # prevent division by zero by setting std to 1 where it's 0

        # Normalize
        x = (x - self.stats['mean']) / self.stats['std']
        x = np.nan_to_num(x, nan=0.0)
        x = np.expand_dims(x.transpose(0, 2, 1), axis=-1)

        return x, labels

    def align_series(self, x: pd.Series, max_len: int) -> np.ndarray:
        '''Pad series to the same length.'''
        x = x.values
        padded_series = np.full(max_len, np.nan)
        # truncate
        if x.shape[0] > max_len:
            padded_series = x[:max_len]
        # pad NaN
        else:
            padded_series[:x.shape[0]] = x
        return padded_series

    def save_data(self, root_path, train_x, train_y, test_x, test_y):
        '''Save the preprocessed data to a binary file.'''

        np.save(os.path.join(root_path, 'train_inputs.npy'), train_x)
        np.save(os.path.join(root_path, 'train_labels.npy'), train_y)
        np.save(os.path.join(root_path, 'test_inputs.npy'), test_x)
        np.save(os.path.join(root_path, 'test_labels.npy'), test_y)


if __name__ == '__main__':
    processor = UEAProcessor()
    processor.run()
