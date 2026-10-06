from .anomaly_metrics import (anomaly_adjusted_f1, anomaly_adjusted_precision,
                              anomaly_adjusted_recall, anomaly_f1,
                              anomaly_precision, anomaly_recall)
from .cls_metrics import accuracy
from .corr import masked_corr
from .huber import masked_huber
from .mae import masked_mae
from .mape import masked_mape
from .metric_meter import AvgMeter, RMSEMeter
from .mse import masked_mse
from .r_square import masked_r2
from .rmse import masked_rmse
from .smape import masked_smape
from .wape import masked_wape

ALL_METRICS = {
            'MAE': masked_mae,
            'MSE': masked_mse,
            'RMSE': masked_rmse,
            'MAPE': masked_mape,
            'WAPE': masked_wape,
            'SMAPE': masked_smape,
            'R2': masked_r2,
            'CORR': masked_corr,
            'HUBER': masked_huber,
            'Accuracy': accuracy,
            'AnomalyPrecision': anomaly_precision,
            'AnomalyRecall': anomaly_recall,
            'AnomalyF1': anomaly_f1,
            'AnomalyAdjustedPrecision': anomaly_adjusted_precision,
            'AnomalyAdjustedRecall': anomaly_adjusted_recall,
            'AnomalyAdjustedF1': anomaly_adjusted_f1,
            }

METRIC_METER = {
    'RMSE': RMSEMeter,
    'default': AvgMeter
}

__all__ = [
    'masked_mae',
    'masked_mse',
    'masked_rmse',
    'incremental_masked_rmse',
    'masked_mape',
    'masked_wape',
    'masked_smape',
    'masked_r2',
    'masked_corr',
    'masked_huber',
    'accuracy',
    'anomaly_precision',
    'anomaly_recall',
    'anomaly_f1',
    'anomaly_adjusted_precision',
    'anomaly_adjusted_recall',
    'anomaly_adjusted_f1',
    'ALL_METRICS',
    'METRIC_METER'
]
