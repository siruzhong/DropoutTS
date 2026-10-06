import torch


def _flatten_scores_labels(
        anomaly_scores: torch.Tensor,
        anomaly_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return anomaly_scores.reshape(-1), anomaly_labels.reshape(-1).bool()


def _binary_stats(
        anomaly_scores: torch.Tensor,
        anomaly_labels: torch.Tensor,
        anomaly_threshold: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores, labels = _flatten_scores_labels(anomaly_scores, anomaly_labels)
    preds = scores >= anomaly_threshold
    tp = (preds & labels).float().sum()
    fp = (preds & ~labels).float().sum()
    fn = (~preds & labels).float().sum()
    return tp, fp, fn


def _point_adjusted_predictions(
        anomaly_scores: torch.Tensor,
        anomaly_labels: torch.Tensor,
        anomaly_threshold: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scores, labels = _flatten_scores_labels(anomaly_scores, anomaly_labels)
    preds = scores >= anomaly_threshold
    adjusted = preds.clone()

    label_cpu = labels.detach().cpu()
    pred_cpu = preds.detach().cpu()
    start = None
    for idx, is_anomaly in enumerate(label_cpu.tolist()):
        if is_anomaly and start is None:
            start = idx
        is_last = idx == len(label_cpu) - 1
        if start is not None and (not is_anomaly or is_last):
            end = idx + 1 if is_anomaly and is_last else idx
            if pred_cpu[start:end].any():
                adjusted[start:end] = True
            start = None
    return adjusted, labels


def _adjusted_binary_stats(
        anomaly_scores: torch.Tensor,
        anomaly_labels: torch.Tensor,
        anomaly_threshold: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    preds, labels = _point_adjusted_predictions(anomaly_scores, anomaly_labels, anomaly_threshold)
    tp = (preds & labels).float().sum()
    fp = (preds & ~labels).float().sum()
    fn = (~preds & labels).float().sum()
    return tp, fp, fn


def anomaly_precision(
        anomaly_scores: torch.Tensor,
        anomaly_labels: torch.Tensor,
        anomaly_threshold: torch.Tensor,
        **kwargs) -> torch.Tensor:
    tp, fp, _ = _binary_stats(anomaly_scores, anomaly_labels, anomaly_threshold)
    return tp / (tp + fp + 1e-8)


def anomaly_recall(
        anomaly_scores: torch.Tensor,
        anomaly_labels: torch.Tensor,
        anomaly_threshold: torch.Tensor,
        **kwargs) -> torch.Tensor:
    tp, _, fn = _binary_stats(anomaly_scores, anomaly_labels, anomaly_threshold)
    return tp / (tp + fn + 1e-8)


def anomaly_f1(
        anomaly_scores: torch.Tensor,
        anomaly_labels: torch.Tensor,
        anomaly_threshold: torch.Tensor,
        **kwargs) -> torch.Tensor:
    precision = anomaly_precision(anomaly_scores, anomaly_labels, anomaly_threshold)
    recall = anomaly_recall(anomaly_scores, anomaly_labels, anomaly_threshold)
    return 2 * precision * recall / (precision + recall + 1e-8)


def anomaly_adjusted_precision(
        anomaly_scores: torch.Tensor,
        anomaly_labels: torch.Tensor,
        anomaly_threshold: torch.Tensor,
        **kwargs) -> torch.Tensor:
    tp, fp, _ = _adjusted_binary_stats(anomaly_scores, anomaly_labels, anomaly_threshold)
    return tp / (tp + fp + 1e-8)


def anomaly_adjusted_recall(
        anomaly_scores: torch.Tensor,
        anomaly_labels: torch.Tensor,
        anomaly_threshold: torch.Tensor,
        **kwargs) -> torch.Tensor:
    tp, _, fn = _adjusted_binary_stats(anomaly_scores, anomaly_labels, anomaly_threshold)
    return tp / (tp + fn + 1e-8)


def anomaly_adjusted_f1(
        anomaly_scores: torch.Tensor,
        anomaly_labels: torch.Tensor,
        anomaly_threshold: torch.Tensor,
        **kwargs) -> torch.Tensor:
    precision = anomaly_adjusted_precision(anomaly_scores, anomaly_labels, anomaly_threshold)
    recall = anomaly_adjusted_recall(anomaly_scores, anomaly_labels, anomaly_threshold)
    return 2 * precision * recall / (precision + recall + 1e-8)
