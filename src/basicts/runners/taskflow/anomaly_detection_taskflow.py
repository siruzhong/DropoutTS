from typing import TYPE_CHECKING, Any, Dict

import torch
from basicts.utils.mask import null_val_mask

from .basicts_taskflow import BasicTSTaskFlow

if TYPE_CHECKING:
    from basicts.runners.basicts_runner import BasicTSRunner


class BasicTSAnomalyDetectionTaskFlow(BasicTSTaskFlow):
    """Reconstruction-based anomaly detection task flow."""

    def preprocess(self, runner: 'BasicTSRunner', data: Dict[str, Any]) -> Dict[str, Any]:
        inputs_mask = null_val_mask(data['inputs'], runner.cfg.null_val)
        targets_mask = null_val_mask(data['targets'], runner.cfg.null_val)

        if runner.scaler is not None:
            data['inputs'] = runner.scaler.transform(data['inputs'], inputs_mask)
            data['targets'] = runner.scaler.transform(data['targets'], targets_mask)

        fill_value = torch.tensor(runner.cfg.null_to_num, device=data['inputs'].device)
        data['inputs'] = torch.where(inputs_mask, data['inputs'], fill_value)
        data['targets'] = torch.where(targets_mask, data['targets'], fill_value)
        data['targets_mask'] = targets_mask
        return data

    def postprocess(self, runner: 'BasicTSRunner', forward_return: Dict[str, Any]) -> Dict[str, Any]:
        if runner.cfg.rescale and runner.scaler is not None:
            forward_return['prediction'] = runner.scaler.inverse_transform(forward_return['prediction'])
            forward_return['targets'] = runner.scaler.inverse_transform(
                forward_return['targets'], forward_return['targets_mask'])

        err = torch.abs(forward_return['prediction'] - forward_return['targets'])
        if err.ndim > 2:
            err = err.mean(dim=tuple(range(2, err.ndim)))
        forward_return['anomaly_scores'] = err
        forward_return['anomaly_labels'] = forward_return.get('labels', torch.zeros_like(err)).float()

        threshold = getattr(runner, "fitted_anomaly_threshold", None)
        if threshold is None:
            threshold = runner.cfg.anomaly_threshold
        if threshold is None:
            q = float(runner.cfg.anomaly_score_percentile) / 100.0
            threshold = torch.quantile(err.detach().flatten(), q)
        forward_return['anomaly_threshold'] = torch.as_tensor(
            threshold, dtype=err.dtype, device=err.device)
        return forward_return

    def get_weight(self, forward_return: Dict[str, Any]) -> float:
        return forward_return['targets_mask'].sum().item()
