"""DropoutTS Callback: Task-Loss-Driven Sample-Adaptive Dropout

This callback integrates DropoutTS into the training pipeline, automatically
replacing standard dropout layers with sample-adaptive dropout and managing
the optimization of RRF parameters.
"""

import os
from typing import TYPE_CHECKING, List, Tuple, Any, Optional, Dict

import torch
from torch import nn

from basicts.modules import DropoutTS, SampleAdaptiveDropout, DropoutTSContext
from .callback import BasicTSCallback

if TYPE_CHECKING:
    from basicts.runners.basicts_runner import BasicTSRunner


def replace_dropout_layers(model: nn.Module) -> List[Tuple[Any, Any, nn.Dropout, SampleAdaptiveDropout]]:
    """Recursively replace all nn.Dropout layers with SampleAdaptiveDropout.
    
    Args:
        model: PyTorch model to process.
    
    Returns:
        List of tuples (parent, key, original_dropout, new_dropout) for each replacement.
    """
    replaced_layers = []
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device('cpu')
    
    def _replace_recursive(module: nn.Module, parent_name: str = "") -> None:
        """Recursively traverse model and replace dropout layers."""
        for name, child in list(module.named_children()):
            full_name = f"{parent_name}.{name}" if parent_name else name
            
            if isinstance(child, nn.Dropout):
                new_dropout = SampleAdaptiveDropout(p=child.p).to(device)
                setattr(module, name, new_dropout)
                replaced_layers.append((module, name, child, new_dropout))
            elif isinstance(child, nn.ModuleList):
                # Handle dropout layers in ModuleList
                for i, submodule in enumerate(child):
                    if isinstance(submodule, nn.Dropout):
                        new_dropout = SampleAdaptiveDropout(p=submodule.p).to(device)
                        child[i] = new_dropout
                        replaced_layers.append((child, i, submodule, new_dropout))
            
            _replace_recursive(child, full_name)
    
    _replace_recursive(model)
    return replaced_layers


class DropoutTSCallback(BasicTSCallback):
    """Callback for sample-adaptive dropout with RRF optimization.
    
    This callback:
    1. Replaces standard dropout layers with SampleAdaptiveDropout
    2. Wraps model forward to compute adaptive dropout rates
    3. Adds RRF parameters to optimizer for end-to-end training
    4. Logs statistics and generates visualizations
    """
    
    def __init__(
        self,
        p_min: float = 0.05,
        p_max: float = 0.5,
        init_alpha: float = 10.0,
        init_sensitivity: float = 1.0,
        init_gate: Optional[float] = None,
        use_gate: bool = False,
        enable_statistics: bool = False,
        enable_visualization: bool = False,
        sparsity_weight: float = 0.0,
        detrend_method: str = 'robust_ols',
        use_instance_norm: bool = True,
        use_sfm_anchor: bool = True
    ):
        """Initialize DropoutTSCallback.
        
        Args:
            p_min: Minimum dropout rate for clean samples.
            p_max: Maximum dropout rate for noisy samples.
            init_alpha: Initial value for RRF sigmoid sharpness parameter.
            init_sensitivity: Initial value for noise score sensitivity parameter.
            init_gate: Initial value for gate parameter (in [0, 1]). If None and use_gate=True, defaults to 0.5.
            use_gate: Whether to use gate mechanism. If False, gate is disabled.
            enable_statistics: If True, log detailed statistics each epoch.
            enable_visualization: If True, generate visualization plots each epoch.
            sparsity_weight: Weight for sparsity regularization loss.
            detrend_method: Detrending method for ablation study ('none', 'simple', 'robust_ols').
            use_instance_norm: Whether to use instance-level normalization for ablation study.
            use_sfm_anchor: Whether to use SFM as physical anchor for ablation study.
        """
        super().__init__()
        self.p_min = p_min
        self.p_max = p_max
        self.init_alpha = init_alpha
        self.init_sensitivity = init_sensitivity
        self.init_gate = init_gate
        self.use_gate = use_gate
        self.enable_statistics = enable_statistics
        self.enable_visualization = enable_visualization
        self.sparsity_weight = sparsity_weight
        
        # Ablation study parameters
        self.detrend_method = detrend_method
        self.use_instance_norm = use_instance_norm
        self.use_sfm_anchor = use_sfm_anchor
        
        self.dropout_ts: Optional[DropoutTS] = None
        self.original_forward = None
        self.model_wrapped = False
        self.replaced_layers: List[Tuple[Any, Any, nn.Dropout, SampleAdaptiveDropout]] = []
        self._last_batch_inputs: Optional[torch.Tensor] = None
        self._rrf_checked = False
        self._sparsity_info: Optional[Dict[str, torch.Tensor]] = None
    
    def on_train_start(self, runner: "BasicTSRunner", **kwargs) -> None:
        """Initialize DropoutTS and wrap model for adaptive dropout."""
        gate_info = f"use_gate: {self.use_gate}"
        if self.use_gate:
            gate_info += f", init_gate: {self.init_gate if self.init_gate is not None else 0.5}"
        
        # Log ablation study configuration
        ablation_info = (
            f"[Ablation Config] detrend_method: {self.detrend_method}, "
            f"use_instance_norm: {self.use_instance_norm}, "
            f"use_sfm_anchor: {self.use_sfm_anchor}"
        )
        
        runner.logger.info(
            f"DropoutTS initialized - p_min: {self.p_min}, p_max: {self.p_max}, "
            f"init_alpha: {self.init_alpha}, init_sensitivity: {self.init_sensitivity}, "
            f"{gate_info}, sparsity_weight: {self.sparsity_weight}."
        )
        runner.logger.info(ablation_info)
        
        self.dropout_ts = DropoutTS(
            seq_len=None,
            p_min=self.p_min,
            p_max=self.p_max,
            init_alpha=self.init_alpha,
            init_sensitivity=self.init_sensitivity,
            init_gate=self.init_gate,
            use_gate=self.use_gate,
            detrend_method=self.detrend_method,
            use_instance_norm=self.use_instance_norm,
            use_sfm_anchor=self.use_sfm_anchor
        ).to(next(runner.model.parameters()).device)
        
        self._wrap_model(runner)
    
    def _wrap_model(self, runner: "BasicTSRunner") -> None:
        """Replace dropout layers and wrap model forward for adaptive dropout."""
        if self.model_wrapped:
            return
        
        model = runner.model.module if isinstance(runner.model, nn.parallel.DistributedDataParallel) else runner.model
        
        # Replace standard dropout layers
        self.replaced_layers = replace_dropout_layers(model)
        if self.replaced_layers:
            runner.logger.info(f"[DropoutTS] Replaced {len(self.replaced_layers)} dropout layer(s).")
        else:
            runner.logger.info("[DropoutTS] No dropout layers found.")
        
        # Save original forward and create wrapped version
        self.original_forward = model.forward
        self._rrf_checked = False
        
        def wrapped_forward(*args, **kwargs):
            """Wrapped forward that computes and applies adaptive dropout rates."""
            inputs = args[0] if args else kwargs.get("inputs", None)
            dropout_rates = None
            self._sparsity_info = None
            
            if model.training and inputs is not None:
                # Store first batch for statistics/visualization
                if (self.enable_statistics or self.enable_visualization) and self._last_batch_inputs is None:
                    self._last_batch_inputs = inputs.detach().clone()
                
                # Compute adaptive dropout rates and sparsity info
                dropout_rates, self._sparsity_info = self.dropout_ts.compute_dropout_rates(
                    inputs,
                    return_sparsity_info=True
                )
                # Set gate value in context (only if gate is enabled)
                gate_val = self.dropout_ts.gate if (self.dropout_ts.use_gate and hasattr(self.dropout_ts, 'gate') and self.dropout_ts.gate is not None) else None
                DropoutTSContext.set_rates(dropout_rates, gate=gate_val)
            
            # Check and add RRF parameters to optimizer (once per training)
            if not self._rrf_checked and self.dropout_ts.noise_scorer is not None and dropout_rates is not None:
                self._setup_rrf_optimizer(runner, inputs)
                self._rrf_checked = True
            
            # Call original forward
            result = self.original_forward(*args, **kwargs)
            DropoutTSContext.clear()
            
            # Ensure result is in expected format
            return {"prediction": result} if isinstance(result, torch.Tensor) else result
        
        model.forward = wrapped_forward
        self.model_wrapped = True
        runner.logger.info("Model wrapped with DropoutTS.")
    
    def _setup_rrf_optimizer(self, runner: "BasicTSRunner", inputs: Optional[torch.Tensor]) -> None:
        """Add RRF parameters to optimizer and log initial state."""
        if self.dropout_ts is None or self.dropout_ts.noise_scorer is None:
            return
        
        rrf_filter = self.dropout_ts.noise_scorer.rrf_filter
        rrf_params = list(rrf_filter.parameters())
        sensitivity_param = self.dropout_ts.sensitivity
        all_params = rrf_params + [sensitivity_param]
        
        # Add gate parameter only if gate is enabled
        gate_param = None
        if self.dropout_ts.use_gate and hasattr(self.dropout_ts, 'gate') and self.dropout_ts.gate is not None:
            gate_param = self.dropout_ts.gate
            all_params.append(gate_param)
        
        # Check which parameters are not yet in optimizer
        optimizer_param_ids = {id(p) for group in runner.optimizer.param_groups for p in group['params']}
        params_to_add = [p for p in all_params if id(p) not in optimizer_param_ids]
        
        if params_to_add:
            runner.optimizer.add_param_group({
                'params': params_to_add,
                'lr': runner.optimizer.param_groups[0]['lr']
            })
            if gate_param is not None:
                runner.logger.info("[DropoutTS] Added RRF params (Alpha, SFM_Scale, SFM_Bias), Sensitivity, and Gate to optimizer.")
            else:
                runner.logger.info("[DropoutTS] Added RRF params (Alpha, SFM_Scale, SFM_Bias) and Sensitivity to optimizer.")
        
        # Log initial parameter values
        alpha_val = rrf_filter.softplus(rrf_filter.alpha).item()
        sfm_scale_val = rrf_filter.sfm_scale.mean().item()
        sfm_bias_val = rrf_filter.sfm_bias.mean().item()
        sensitivity_val = torch.nn.functional.softplus(self.dropout_ts.sensitivity).item()
        gate_val = None
        if self.dropout_ts.use_gate and hasattr(self.dropout_ts, 'gate') and self.dropout_ts.gate is not None:
            gate_val = torch.sigmoid(self.dropout_ts.gate).item()
        
        if inputs is not None:
            # Get initial SFM score and dynamic threshold from first batch
            was_training = runner.model.training
            runner.model.eval()
            details = self.dropout_ts.noise_scorer(
                inputs, 
                return_details=True
            )
            sfm_score_val = details['sfm_score'].mean().item()
            dynamic_threshold_val = details['dynamic_threshold'].mean().item()
            if was_training:
                runner.model.train()
            
            gate_str = f", Gate={gate_val:.4f}" if gate_val is not None else ""
            runner.logger.info(
                f"[DropoutTS] Initial State: Alpha={alpha_val:.4f}, SFM_Scale={sfm_scale_val:.4f}, SFM_Bias={sfm_bias_val:.4f}, "
                f"SFM_Score={sfm_score_val:.4f}, Dynamic_Threshold={dynamic_threshold_val:.4f}, "
                f"Sensitivity={sensitivity_val:.4f}{gate_str}"
            )
        else:
            gate_str = f", Gate={gate_val:.4f}" if gate_val is not None else ""
            runner.logger.info(
                f"[DropoutTS] Initial State: Alpha={alpha_val:.4f}, SFM_Scale={sfm_scale_val:.4f}, SFM_Bias={sfm_bias_val:.4f}, "
                f"Sensitivity={sensitivity_val:.4f}{gate_str}"
            )
    
    def on_epoch_start(self, runner: "BasicTSRunner", **kwargs) -> None:
        """Reset batch inputs at start of each epoch."""
        self._last_batch_inputs = None
        self._sparsity_info = None
    
    def on_compute_loss(self, runner: "BasicTSRunner", **kwargs) -> None:
        """Add sparsity regularization loss to the main loss."""
        if self._sparsity_info is None:
            return
        
        forward_return = kwargs.get("forward_return", {})
        if not isinstance(forward_return, dict):
            return
        
        # Compute sparsity regularization loss
        sparsity_loss_mask = self._sparsity_info.get('sparsity_loss_mask', torch.tensor(0.0))
        sparsity_loss = self.sparsity_weight * sparsity_loss_mask
        
        # Store sparsity loss in forward_return for logging
        forward_return['sparsity_loss'] = sparsity_loss
        
        # Compute main loss first (this will be used by _metric_forward)
        # Then add sparsity loss and set it back to forward_return['loss']
        # This ensures the total loss (main + sparsity) is used for backward pass
        main_loss = runner._metric_forward(runner.loss, forward_return)
        total_loss = main_loss + sparsity_loss
        forward_return['loss'] = total_loss
    
    def on_epoch_end(self, runner: "BasicTSRunner", **kwargs) -> None:
        """Log detailed statistics and generate visualizations."""
        if self.dropout_ts is None or self.dropout_ts.noise_scorer is None:
            return
        
        epoch = kwargs.get('epoch', runner.epoch if hasattr(runner, 'epoch') else 0)
        rrf = self.dropout_ts.noise_scorer.rrf_filter
        
        # Get current parameter values
        alpha_val = rrf.softplus(rrf.alpha).item()
        sfm_scale_val = rrf.sfm_scale.mean().item()
        sfm_bias_val = rrf.sfm_bias.mean().item()
        sensitivity_val = torch.nn.functional.softplus(self.dropout_ts.sensitivity).item()
        gate_val = None
        if self.dropout_ts.use_gate and hasattr(self.dropout_ts, 'gate') and self.dropout_ts.gate is not None:
            gate_val = torch.sigmoid(self.dropout_ts.gate).item()
        
        # Get SFM score and dynamic threshold from last batch
        if self._last_batch_inputs is not None:
            was_training = self.dropout_ts.training
            self.dropout_ts.eval()
            details = self.dropout_ts.noise_scorer(
                self._last_batch_inputs, 
                return_details=True
            )
            sfm_score_val = details['sfm_score'].mean().item()
            dynamic_threshold_val = details['dynamic_threshold'].mean().item()
            
            # Get dropout rates and noise scores for each sample
            dropout_rates, _ = self.dropout_ts.compute_dropout_rates(
                self._last_batch_inputs,
                return_sparsity_info=True
            )
            noise_scores = details['noise_scores']
            
            # Get sparsity information if available
            sparsity_info_str = ""
            if self._sparsity_info is not None:
                sparsity_loss_mask_val = self._sparsity_info.get('sparsity_loss_mask', torch.tensor(0.0)).item()
                total_sparsity_loss_val = self.sparsity_weight * sparsity_loss_mask_val
                sparsity_info_str = (
                    f", SparsityLossMask={sparsity_loss_mask_val:.6f}, "
                    f"TotalSparsityLoss={total_sparsity_loss_val:.6f}"
                )
            
            if was_training:
                self.dropout_ts.train()
            
            gate_str = f", Gate={gate_val:.4f}" if gate_val is not None else ""
            runner.logger.info(
                f"[DropoutTS] Epoch {epoch}: Alpha={alpha_val:.4f}, SFM_Scale={sfm_scale_val:.4f}, SFM_Bias={sfm_bias_val:.4f}, "
                f"SFM_Score={sfm_score_val:.4f}, Dynamic_Threshold={dynamic_threshold_val:.4f}, "
                f"Sensitivity={sensitivity_val:.4f}{gate_str}{sparsity_info_str}"
            )
        else:
            gate_str = f", Gate={gate_val:.4f}" if gate_val is not None else ""
            runner.logger.info(
                f"[DropoutTS] Epoch {epoch}: Alpha={alpha_val:.4f}, SFM_Scale={sfm_scale_val:.4f}, SFM_Bias={sfm_bias_val:.4f}, "
                f"Sensitivity={sensitivity_val:.4f}{gate_str}"
            )
        
        # Generate statistics and visualizations if enabled
        if self._last_batch_inputs is not None and (self.enable_statistics or self.enable_visualization):
            self._log_statistics_and_visualize(runner, epoch)
    
    def _log_statistics_and_visualize(self, runner: "BasicTSRunner", epoch: int) -> None:
        """Log detailed statistics and generate visualizations."""
        was_training = self.dropout_ts.training
        self.dropout_ts.eval()
        
        dropout_rates, _ = self.dropout_ts.compute_dropout_rates(
            self._last_batch_inputs, 
            visualize=False,
            return_sparsity_info=True
        )
        
        details = self.dropout_ts.noise_scorer(
            self._last_batch_inputs, 
            return_details=True
        )
        noise_scores = details['noise_scores']
        
        if was_training:
            self.dropout_ts.train()
        
        # Log statistics
        if self.enable_statistics:
            noise_mean = noise_scores.mean().item()
            noise_std = noise_scores.std().item()
            noise_min = noise_scores.min().item()
            noise_max = noise_scores.max().item()
            
            dropout_mean = dropout_rates.mean().item()
            dropout_std = dropout_rates.std().item()
            dropout_min = dropout_rates.min().item()
            dropout_max = dropout_rates.max().item()
            
            stat_msg = (
                f"[DropoutTS] Epoch {epoch} Statistics - "
                f"Noise (MAE): mean={noise_mean:.4f}, std={noise_std:.4f}, "
                f"min={noise_min:.4f}, max={noise_max:.4f} | "
                f"Dropout: mean={dropout_mean:.4f}, std={dropout_std:.4f}, "
                f"min={dropout_min:.4f}, max={dropout_max:.4f}"
            )
            runner.logger.info(stat_msg)
        
        # Generate visualization
        if self.enable_visualization:
            original_dir = os.getcwd()
            os.chdir(runner.ckpt_save_dir)
            self.dropout_ts.compute_dropout_rates(
                self._last_batch_inputs, 
                visualize=True, 
                max_samples=6
            )
            os.chdir(original_dir)
            runner.logger.info(
                f"[DropoutTS] Visualization saved to: "
                f"{os.path.join(runner.ckpt_save_dir, 'noise_scoring_visualization.png')}"
            )
    
    def on_train_end(self, runner: "BasicTSRunner", **kwargs) -> None:
        """Restore model and log final state."""
        # Log final parameter values
        if self.dropout_ts is not None and self.dropout_ts.noise_scorer is not None:
            rrf = self.dropout_ts.noise_scorer.rrf_filter
            alpha_val = rrf.softplus(rrf.alpha).item()
            sfm_scale_val = rrf.sfm_scale.mean().item()
            sfm_bias_val = rrf.sfm_bias.mean().item()
            sensitivity_val = torch.nn.functional.softplus(self.dropout_ts.sensitivity).item()
            gate_val = None
            if self.dropout_ts.use_gate and hasattr(self.dropout_ts, 'gate') and self.dropout_ts.gate is not None:
                gate_val = torch.sigmoid(self.dropout_ts.gate).item()
            
            if self._last_batch_inputs is not None:
                was_training = self.dropout_ts.training
                self.dropout_ts.eval()
                details = self.dropout_ts.noise_scorer(
                    self._last_batch_inputs, 
                    return_details=True
                )
                sfm_score_val = details['sfm_score'].mean().item()
                dynamic_threshold_val = details['dynamic_threshold'].mean().item()
                if was_training:
                    self.dropout_ts.train()
                
                gate_str = f", Gate={gate_val:.4f}" if gate_val is not None else ""
                runner.logger.info(
                    f"[DropoutTS] Finished. Final Params: Alpha={alpha_val:.4f}, "
                    f"SFM_Scale={sfm_scale_val:.4f}, SFM_Bias={sfm_bias_val:.4f}, SFM_Score={sfm_score_val:.4f}, "
                    f"Dynamic_Threshold={dynamic_threshold_val:.4f}, Sensitivity={sensitivity_val:.4f}{gate_str}"
                )
            else:
                gate_str = f", Gate={gate_val:.4f}" if gate_val is not None else ""
                runner.logger.info(
                    f"[DropoutTS] Finished. Final Params: Alpha={alpha_val:.4f}, "
                    f"SFM_Scale={sfm_scale_val:.4f}, SFM_Bias={sfm_bias_val:.4f}, Sensitivity={sensitivity_val:.4f}{gate_str}"
                )
        
        # Restore original dropout layers
        if self.replaced_layers:
            model = runner.model.module if isinstance(runner.model, nn.parallel.DistributedDataParallel) else runner.model
            for parent, key, original_dropout, _ in self.replaced_layers:
                if isinstance(key, int):
                    parent[key] = original_dropout
                else:
                    setattr(parent, key, original_dropout)
            self.replaced_layers = []
        
        # Restore original forward
        if self.model_wrapped and self.original_forward is not None:
            model = runner.model.module if isinstance(runner.model, nn.parallel.DistributedDataParallel) else runner.model
            model.forward = self.original_forward
            self.model_wrapped = False
            runner.logger.info("Restored original model forward.")
        
        DropoutTSContext.clear()
