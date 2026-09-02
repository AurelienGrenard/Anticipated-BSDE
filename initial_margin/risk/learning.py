"""Two-step conditional VaR/expected-shortfall learning estimator."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from hermite import HermiteModel

from ..config import ExperimentConfig
from .common import RiskEstimate, empirical_upper_cvar, project_future_value
from .config import RiskEstimatorConfig


class SoftplusNetwork(nn.Module):
    """Represent one scalar conditional quantile by a compact neural network."""

    def __init__(self, width: int, hidden_layers: int) -> None:
        """Build equal-width Softplus hidden layers and one linear output."""
        super().__init__()
        layers: list[nn.Module] = []
        input_width = 1
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(input_width, width), nn.Softplus()))
            input_width = width
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(width, 1)

    def features(self, state: torch.Tensor) -> torch.Tensor:
        """Return the last hidden representation reused by the ES regression."""
        return self.hidden(state[:, None])

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Evaluate the learned conditional VaR."""
        return self.output(self.features(state)).squeeze(1)


@dataclass(frozen=True)
class LearningDiagnostics:
    """Summarize VaR calibration and the frozen-feature ES regression."""

    updates: int
    quantile_loss: float
    exceedance_rate: float
    es_gram_condition: float


class ConditionalVarEsLearner:
    """Warm-start VaR across dates and infer ES by a linear output regression."""

    def __init__(
        self,
        settings: RiskEstimatorConfig,
        alpha: float,
        device: torch.device,
        seed: int,
    ) -> None:
        """Initialize the network, normalization state, and deterministic seeds."""
        self.settings = settings
        self.alpha = alpha
        self.device = device
        self.seed = seed
        self.dtype = (
            torch.float32
            if settings.learning_dtype == "float32"
            else torch.float64
        )
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        self.network = SoftplusNetwork(
            settings.learning_width,
            settings.learning_hidden_layers,
        ).to(device=device, dtype=self.dtype)
        self._previous_mean: torch.Tensor | None = None
        self._previous_scale: torch.Tensor | None = None
        self._first_fit = True

    def _adapt_output_normalization(
        self,
        new_mean: torch.Tensor,
        new_scale: torch.Tensor,
    ) -> None:
        """Preserve the raw VaR prediction when target normalization changes."""
        if self._previous_mean is None or self._previous_scale is None:
            return
        ratio = self._previous_scale / new_scale
        with torch.no_grad():
            self.network.output.weight.mul_(ratio)
            self.network.output.bias.mul_(self._previous_scale).add_(
                self._previous_mean - new_mean
            ).div_(new_scale)

    def _train_var(
        self,
        state: torch.Tensor,
        target: torch.Tensor,
        step: int,
    ) -> tuple[torch.Tensor, int]:
        """Minimize the Rockafellar quantile objective by Adam."""
        updates = (
            self.settings.learning_initial_updates
            if self._first_fit
            else self.settings.learning_updates_per_date
        )
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.settings.learning_rate,
        )
        generator = torch.Generator(device=self.device).manual_seed(
            self.seed + 10_007 * (step + 1)
        )
        batch_size = min(self.settings.learning_batch_size, state.numel())
        tail_probability = 1.0 - self.alpha
        for _ in range(updates):
            indices = torch.randint(
                state.numel(),
                (batch_size,),
                generator=generator,
                device=self.device,
            )
            quantile = self.network(state[indices])
            loss = (
                torch.relu(target[indices] - quantile)
                + tail_probability * quantile
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        self._first_fit = False
        with torch.no_grad():
            quantile = self.network(state)
        return quantile, updates

    def _fit_es(
        self,
        state: torch.Tensor,
        target: torch.Tensor,
        quantile: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """Freeze VaR features and solve the second-stage ES least squares."""
        with torch.no_grad():
            regression_target = quantile + torch.relu(
                target - quantile
            ) / (1.0 - self.alpha)
            features = self.network.features(state)
            feature_mean = features.mean(dim=0, keepdim=True)
            feature_scale = features.std(dim=0, keepdim=True).clamp_min(1e-5)
            normalized = (features - feature_mean) / feature_scale
            centered_target = regression_target - regression_target.mean()
            sample_count = normalized.shape[0]
            gram = normalized.T @ normalized / sample_count
            rhs = normalized.T @ centered_target[:, None] / sample_count
            singular_values = torch.linalg.svdvals(gram.double())
            smallest = singular_values.min().clamp_min(1e-30)
            condition = float(singular_values.max() / smallest)
            penalty = self.settings.learning_es_ridge * torch.eye(
                gram.shape[0],
                device=self.device,
                dtype=self.dtype,
            )
            coefficients = torch.linalg.solve(gram + penalty, rhs)
            fitted = (
                normalized @ coefficients
            ).squeeze(1) + regression_target.mean()
        return fitted, condition

    def fit_predict(
        self,
        state: torch.Tensor,
        loss_gap: torch.Tensor,
        step: int,
    ) -> tuple[torch.Tensor, LearningDiagnostics]:
        """Learn conditional VaR then ES and evaluate ES on training states."""
        state_learning = state.to(dtype=self.dtype)
        target_learning = loss_gap.to(dtype=self.dtype)
        target_mean = target_learning.mean()
        target_scale = target_learning.std().clamp_min(1e-6)
        self._adapt_output_normalization(target_mean, target_scale)
        standardized = (target_learning - target_mean) / target_scale
        quantile, updates = self._train_var(state_learning, standardized, step)
        es_standardized, condition = self._fit_es(
            state_learning,
            standardized,
            quantile,
        )
        with torch.no_grad():
            loss = (
                torch.relu(standardized - quantile)
                + (1.0 - self.alpha) * quantile
            ).mean()
            exceedance = (standardized > quantile).float().mean()
        self._previous_mean = target_mean.detach()
        self._previous_scale = target_scale.detach()
        prediction = es_standardized * target_scale + target_mean
        return prediction.to(dtype=state.dtype), LearningDiagnostics(
            updates=updates,
            quantile_loss=float(loss),
            exceedance_rate=float(exceedance),
            es_gram_condition=condition,
        )


class LearningRiskEstimator:
    """Learn conditional VaR and ES from delayed outer-path losses."""

    name = "learning"

    def __init__(
        self,
        settings: RiskEstimatorConfig,
        alpha: float,
        device: torch.device,
        seed: int,
    ) -> None:
        """Create the date-to-date warm-started two-stage learner."""
        self.learner = ConditionalVarEsLearner(settings, alpha, device, seed)

    def estimate(
        self,
        y_models: list[HermiteModel],
        y_values: torch.Tensor,
        step: int,
        brownian_now: torch.Tensor,
        spot_now: torch.Tensor,
        state: torch.Tensor,
        current_bs: torch.Tensor,
        future_spot_path: torch.Tensor,
        config: ExperimentConfig,
    ) -> RiskEstimate:
        """Fit VaR then ES from one realized delayed loss per outer path."""
        del y_models, brownian_now, spot_now
        future_step = min(step + config.delay_steps, config.time_steps)
        future_value = project_future_value(
            y_values[:, future_step],
            future_spot_path,
            config,
        )
        loss_gap = current_bs - future_value
        if step == 0:
            value = empirical_upper_cvar(loss_gap, config.alpha)
            return RiskEstimate(
                value.expand_as(state),
                {
                    "updates": 0,
                    "quantile_loss": 0.0,
                    "exceedance_rate": 1.0 - config.alpha,
                    "es_gram_condition": 1.0,
                },
            )
        prediction, diagnostics = self.learner.fit_predict(state, loss_gap, step)
        return RiskEstimate(prediction, asdict(diagnostics))
