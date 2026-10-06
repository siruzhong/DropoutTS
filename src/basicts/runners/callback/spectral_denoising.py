"""SpectralDenoisingCallback: Input-Level Spectral Denoising Baseline

This callback serves as an ablation baseline for DropoutTS. Instead of mapping
spectral noise scores to adaptive dropout rates, it replaces the model's raw
input with the spectrally-reconstructed (denoised) signal x̂ during training.

The RRF filter parameters (alpha, sfm_scale, sfm_bias) are still learned
end-to-end via task loss gradients, making this a fair comparison that isolates
the contribution of the adaptive dropout mechanism over simple input denoising.
"""

from typing import TYPE_CHECKING, Optional, Dict, Any

import torch
from torch import nn

from basicts.modules import DropoutTS
from .callback import BasicTSCallback

if TYPE_CHECKING:
    from basicts.runners.basicts_runner import BasicTSRunner


class SpectralDenoisingCallback(BasicTSCallback):
    """Spectral denoising baseline: replace model input with spectrally-denoised signal.

    This callback provides a direct ablation against DropoutTS. It uses the same
    SFM-anchored RRF filter to compute the denoised reconstruction x̂ of each input
    sample, then feeds x̂ to the backbone instead of the original noisy x.

    Unlike DropoutTSCallback:
    - Dropout layers are NOT replaced (fixed dropout rates apply as usual).
    - No noise score → dropout rate mapping is performed.
    - Only the RRF filter params (alpha, sfm_scale, sfm_bias) are added to the
      optimizer; no sensitivity or gate parameters.

    The filter is still learned end-to-end: task loss gradients flow back through
    the differentiable FFT → mask → IFFT denoising pipeline.
    """

    def __init__(
        self,
        init_alpha: float = 10.0,
        detrend_method: str = 'robust_ols',
        use_instance_norm: bool = True,
        use_sfm_anchor: bool = True,
        sparsity_weight: float = 0.0,
    ):
        """Initialize SpectralDenoisingCallback.

        Args:
            init_alpha: Initial value for RRF sigmoid sharpness parameter.
            detrend_method: Detrending method ('none', 'simple', 'robust_ols').
            use_instance_norm: Whether to use instance-level normalization for FFT amplitudes.
            use_sfm_anchor: Whether to use SFM as physical anchor in RRF filter.
            sparsity_weight: Weight for optional sparsity regularization on the mask.
        """
        super().__init__()
        self.init_alpha = init_alpha
        self.detrend_method = detrend_method
        self.use_instance_norm = use_instance_norm
        self.use_sfm_anchor = use_sfm_anchor
        self.sparsity_weight = sparsity_weight

        self.dropout_ts: Optional[DropoutTS] = None
        self.original_forward = None
        self.model_wrapped = False
        self._rrf_checked = False
        self._sparsity_info: Optional[Dict[str, Any]] = None
        self._input_was_transposed: bool = False

    def on_train_start(self, runner: "BasicTSRunner", **kwargs) -> None:
        """Initialize the noise scorer and wrap the model forward."""
        runner.logger.info(
            f"[SpectralDenoising] Initialized - init_alpha: {self.init_alpha}, "
            f"detrend_method: {self.detrend_method}, use_instance_norm: {self.use_instance_norm}, "
            f"use_sfm_anchor: {self.use_sfm_anchor}, sparsity_weight: {self.sparsity_weight}"
        )
        # p_min = p_max = 0 means the DropoutTS module is used solely to host the
        # noise_scorer; the dropout rate computation path is never exercised.
        self.dropout_ts = DropoutTS(
            seq_len=None,
            p_min=0.0,
            p_max=0.0,
            init_alpha=self.init_alpha,
            use_gate=False,
            detrend_method=self.detrend_method,
            use_instance_norm=self.use_instance_norm,
            use_sfm_anchor=self.use_sfm_anchor,
        ).to(next(runner.model.parameters()).device)
        self._wrap_model(runner)

    def _wrap_model(self, runner: "BasicTSRunner") -> None:
        """Wrap model forward to substitute inputs with denoised versions."""
        if self.model_wrapped:
            return

        model = (
            runner.model.module
            if isinstance(runner.model, nn.parallel.DistributedDataParallel)
            else runner.model
        )

        self.original_forward = model.forward
        self._rrf_checked = False

        def wrapped_forward(*args, **kwargs):
            inputs = args[0] if args else kwargs.get("inputs", None)
            self._sparsity_info = None

            if model.training and inputs is not None:
                # Lazy-initialize the noise scorer on the first batch
                if self.dropout_ts.noise_scorer is None:
                    # Trigger lazy init by calling compute_dropout_rates once
                    self.dropout_ts.compute_dropout_rates(inputs)

                # Get the spectrally-denoised reconstruction
                details = self.dropout_ts.noise_scorer(inputs, return_details=True)

                # 'reconstructed' is always [B, C, T] (noise scorer internal format).
                # Most backbones expect [B, T, C], so we need to reverse the permutation
                # that NoiseScorer applied when it received a [B, T, C] input.
                reconstructed = details['reconstructed']  # [B, C, T]
                if inputs.dim() == 3 and inputs.shape[1] != inputs.shape[2]:
                    # Original input was [B, T, C] → transpose reconstructed back
                    denoised = reconstructed.transpose(1, 2)   # [B, T, C]
                else:
                    # Original input was already [B, C, T] (or square), keep as-is
                    denoised = reconstructed

                self._sparsity_info = {
                    'sparsity_loss_mask': details.get(
                        'sparsity_loss_mask',
                        torch.tensor(0.0, device=inputs.device)
                    )
                }

                # Replace the first positional arg (inputs) with the denoised signal;
                # all other args (e.g. timestamps) are passed through unchanged.
                if args:
                    args = (denoised,) + args[1:]
                else:
                    kwargs["inputs"] = denoised

                # Add RRF params to optimizer on the first training step
                if not self._rrf_checked:
                    self._setup_rrf_optimizer(runner)
                    self._rrf_checked = True

            result = self.original_forward(*args, **kwargs)
            return {"prediction": result} if isinstance(result, torch.Tensor) else result

        model.forward = wrapped_forward
        self.model_wrapped = True
        runner.logger.info("[SpectralDenoising] Model wrapped for input-level denoising.")

    def _setup_rrf_optimizer(self, runner: "BasicTSRunner") -> None:
        """Add RRF filter parameters to the optimizer."""
        if self.dropout_ts is None or self.dropout_ts.noise_scorer is None:
            return

        rrf_filter = self.dropout_ts.noise_scorer.rrf_filter
        rrf_params = list(rrf_filter.parameters())

        optimizer_param_ids = {
            id(p) for group in runner.optimizer.param_groups for p in group['params']
        }
        params_to_add = [p for p in rrf_params if id(p) not in optimizer_param_ids]

        if params_to_add:
            runner.optimizer.add_param_group({
                'params': params_to_add,
                'lr': runner.optimizer.param_groups[0]['lr'],
            })
            runner.logger.info(
                "[SpectralDenoising] Added RRF params (Alpha, SFM_Scale, SFM_Bias) to optimizer."
            )

        # Log initial filter state
        alpha_val = rrf_filter.softplus(rrf_filter.alpha).item()
        sfm_scale_val = rrf_filter.sfm_scale.mean().item()
        sfm_bias_val = rrf_filter.sfm_bias.mean().item()
        runner.logger.info(
            f"[SpectralDenoising] Initial RRF state: Alpha={alpha_val:.4f}, "
            f"SFM_Scale={sfm_scale_val:.4f}, SFM_Bias={sfm_bias_val:.4f}"
        )

    def on_compute_loss(self, runner: "BasicTSRunner", **kwargs) -> None:
        """Optionally add sparsity regularization to the main loss."""
        if self._sparsity_info is None or self.sparsity_weight == 0.0:
            return

        forward_return = kwargs.get("forward_return", {})
        if not isinstance(forward_return, dict):
            return

        sparsity_loss_mask = self._sparsity_info.get('sparsity_loss_mask', torch.tensor(0.0))
        sparsity_loss = self.sparsity_weight * sparsity_loss_mask
        forward_return['sparsity_loss'] = sparsity_loss

        main_loss = runner._metric_forward(runner.loss, forward_return)
        total_loss = main_loss + sparsity_loss
        forward_return['loss'] = total_loss

    def on_epoch_end(self, runner: "BasicTSRunner", **kwargs) -> None:
        """Log current RRF filter state."""
        if self.dropout_ts is None or self.dropout_ts.noise_scorer is None:
            return

        epoch = kwargs.get('epoch', runner.epoch if hasattr(runner, 'epoch') else 0)
        rrf = self.dropout_ts.noise_scorer.rrf_filter
        alpha_val = rrf.softplus(rrf.alpha).item()
        sfm_scale_val = rrf.sfm_scale.mean().item()
        sfm_bias_val = rrf.sfm_bias.mean().item()
        runner.logger.info(
            f"[SpectralDenoising] Epoch {epoch}: Alpha={alpha_val:.4f}, "
            f"SFM_Scale={sfm_scale_val:.4f}, SFM_Bias={sfm_bias_val:.4f}"
        )

    def on_train_end(self, runner: "BasicTSRunner", **kwargs) -> None:
        """Restore original model forward and log final filter state."""
        if self.dropout_ts is not None and self.dropout_ts.noise_scorer is not None:
            rrf = self.dropout_ts.noise_scorer.rrf_filter
            alpha_val = rrf.softplus(rrf.alpha).item()
            sfm_scale_val = rrf.sfm_scale.mean().item()
            sfm_bias_val = rrf.sfm_bias.mean().item()
            runner.logger.info(
                f"[SpectralDenoising] Finished. Final RRF: Alpha={alpha_val:.4f}, "
                f"SFM_Scale={sfm_scale_val:.4f}, SFM_Bias={sfm_bias_val:.4f}"
            )

        if self.model_wrapped and self.original_forward is not None:
            model = (
                runner.model.module
                if isinstance(runner.model, nn.parallel.DistributedDataParallel)
                else runner.model
            )
            model.forward = self.original_forward
            self.model_wrapped = False
            runner.logger.info("[SpectralDenoising] Restored original model forward.")
