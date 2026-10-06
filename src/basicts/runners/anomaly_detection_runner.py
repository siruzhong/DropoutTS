from typing import Any, Dict, Tuple

import torch
from basicts.utils import BasicTSMode
from tqdm import tqdm

from .basicts_runner import BasicTSRunner
from .builder import Builder


class BasicTSAnomalyDetectionRunner(BasicTSRunner):
    """Runner for reconstruction-based time series anomaly detection.

    The base runner computes metrics batch by batch. For windowed anomaly
    detection, that duplicates timestamps and produces unstable thresholds.
    This runner aggregates overlapping window scores back to the original
    timeline, fits one validation-based threshold, and then reports point-level
    precision/recall/F1 on the aggregated sequence.
    """

    fitted_anomaly_threshold: float = None

    def _eval_loop(self, mode: BasicTSMode):
        leave = not (self.training_unit == "step" and mode != BasicTSMode.EVAL)
        if mode == BasicTSMode.VAL:
            data_loader = self.val_data_loader
            meter_type = "val"
        else:
            data_loader = self.test_data_loader
            meter_type = "test"
            if self.fitted_anomaly_threshold is None and self.cfg.anomaly_threshold is None:
                self._ensure_validation_threshold()

        data_iter = tqdm(data_loader, leave=leave)
        score_sum, score_count, labels = self._make_sequence_buffers(data_loader)

        for step, data in enumerate(data_iter):
            data = self.taskflow.preprocess(self, data)
            forward_return = self._forward(self.model, data, step=step)
            self.callback_handler.trigger("on_compute_loss", self, forward_return=forward_return)

            loss = self._metric_forward(self.loss, forward_return)
            loss_weight = self.taskflow.get_weight(forward_return)
            self.update_meter(f"{meter_type}/loss", loss.item(), loss_weight)

            forward_return = self.taskflow.postprocess(self, forward_return)
            if mode == BasicTSMode.EVAL:
                self._save_results(step, forward_return)

            self._accumulate_sequence_scores(score_sum, score_count, labels, forward_return)

        sequence_scores, sequence_labels = self._finalize_sequence_scores(score_sum, score_count, labels)
        if mode == BasicTSMode.VAL and self.cfg.anomaly_threshold is None:
            self.fitted_anomaly_threshold = self._threshold_from_scores(sequence_scores)
            self.logger.info(
                f"Fitted anomaly threshold from validation scores: {self.fitted_anomaly_threshold:.6f}"
            )

        threshold = self.cfg.anomaly_threshold
        if threshold is None:
            threshold = self.fitted_anomaly_threshold
        if threshold is None:
            threshold = self._threshold_from_scores(sequence_scores)

        metric_return = {
            "prediction": sequence_scores,
            "targets": sequence_labels,
            "anomaly_scores": sequence_scores,
            "anomaly_labels": sequence_labels,
            "anomaly_threshold": torch.as_tensor(threshold, dtype=sequence_scores.dtype),
        }
        metric_weight = int(sequence_scores.numel())
        for metric_name, metric_fn in self.metrics.items():
            metric_value = self._metric_forward(metric_fn, metric_return)
            self.update_meter(f"{meter_type}/{metric_name}", metric_value.item(), metric_weight)

    def _ensure_validation_threshold(self) -> None:
        if not self.is_val_initialized:
            self._init_validation()
            self.is_val_initialized = True

        score_sum, score_count, labels = self._make_sequence_buffers(self.val_data_loader)
        for step, data in enumerate(self.val_data_loader):
            data = self.taskflow.preprocess(self, data)
            forward_return = self._forward(self.model, data, step=step)
            forward_return = self.taskflow.postprocess(self, forward_return)
            self._accumulate_sequence_scores(score_sum, score_count, labels, forward_return)

        sequence_scores, _ = self._finalize_sequence_scores(score_sum, score_count, labels)
        self.fitted_anomaly_threshold = self._threshold_from_scores(sequence_scores)
        self.logger.info(
            f"Fitted anomaly threshold from validation scores: {self.fitted_anomaly_threshold:.6f}"
        )

    def _make_sequence_buffers(self, data_loader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dataset = Builder.unwrap_dataset(data_loader.dataset)
        sequence_len = len(dataset.data)
        score_sum = torch.zeros(sequence_len, dtype=torch.float64)
        score_count = torch.zeros(sequence_len, dtype=torch.float64)
        labels = torch.zeros(sequence_len, dtype=torch.float32)
        return score_sum, score_count, labels

    def _accumulate_sequence_scores(
            self,
            score_sum: torch.Tensor,
            score_count: torch.Tensor,
            labels: torch.Tensor,
            forward_return: Dict[str, Any]) -> None:
        scores = forward_return["anomaly_scores"].detach().cpu().float()
        batch_labels = forward_return["anomaly_labels"].detach().cpu().float()
        indices = forward_return["index"].detach().cpu().long()

        for i, start in enumerate(indices.tolist()):
            end = min(start + scores.shape[1], score_sum.shape[0])
            width = end - start
            score_sum[start:end] += scores[i, :width].double()
            score_count[start:end] += 1.0
            labels[start:end] = torch.maximum(labels[start:end], batch_labels[i, :width])

    @staticmethod
    def _finalize_sequence_scores(
            score_sum: torch.Tensor,
            score_count: torch.Tensor,
            labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        valid = score_count > 0
        sequence_scores = torch.zeros_like(score_sum, dtype=torch.float32)
        sequence_scores[valid] = (score_sum[valid] / score_count[valid]).float()
        return sequence_scores[valid], labels[valid]

    def _threshold_from_scores(self, sequence_scores: torch.Tensor) -> float:
        q = float(self.cfg.anomaly_score_percentile) / 100.0
        return float(torch.quantile(sequence_scores.detach().float(), q).item())
