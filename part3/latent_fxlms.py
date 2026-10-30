"""Decoder-Jacobian FxLMS updates for the Part 3 controller."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


def _decode(bundle: Any, z: torch.Tensor) -> torch.Tensor:
    w = bundle.decode(z.unsqueeze(0))
    if isinstance(w, (tuple, list)):
        w = w[0]
    return w.reshape(-1)


def latent_fxlms_update(
    bundle: Any,
    z: torch.Tensor,
    filtered_x: torch.Tensor,
    error: torch.Tensor,
    normalizer: torch.Tensor | None = None,
    *,
    step_size: float,
    epsilon: float = 1e-6,
    max_update_norm: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply one block update matching the Part 1 normalized FxLMS rule.

    `filtered_x` contains one secondary-path-filtered reference tap vector per
    sample (shape B x 512). For each sample, the filter-space gradient is
    weighted by the same `pw[n] = sum(filtered_x[n]**2) + epsilon` used by
    `fxlms.py`. The gradient is projected to the latent coordinates with the
    vector-Jacobian product J(z)^T grad_w.

    `normalizer` may be supplied from Part 1's precomputed `pw` trace. If it is
    omitted, the squared row norm is used. Clipping is off by default to keep
    the update equation explicit; enable it only for a separately reported
    stability experiment.
    """
    if z.ndim != 1:
        raise ValueError("z must have shape (latent_dim,)")
    if step_size <= 0 or epsilon <= 0:
        raise ValueError("step_size and epsilon must be positive")
    x = torch.as_tensor(filtered_x, dtype=z.dtype, device=z.device)
    e = torch.as_tensor(error, dtype=z.dtype, device=z.device).reshape(-1)
    if x.ndim != 2 or x.shape[0] != e.numel():
        raise ValueError("filtered_x must be (block_samples, taps) and match error")
    if normalizer is None:
        pw = x.square().sum(dim=1) + epsilon
    else:
        pw = torch.as_tensor(normalizer, dtype=z.dtype, device=z.device).reshape(-1)
        if pw.numel() != e.numel() or torch.any(pw <= 0):
            raise ValueError("normalizer must be positive with one value per sample")

    # The exact baseline block-normalized gradient, averaged over B samples.
    grad_w = x.transpose(0, 1) @ (e / pw) / max(1, x.shape[0])
    z_leaf = z.detach().clone().requires_grad_(True)
    w = _decode(bundle, z_leaf)
    if w.numel() != x.shape[1]:
        raise ValueError(f"Decoder returned {w.numel()} taps; expected {x.shape[1]}")
    grad_z = torch.autograd.grad(w, z_leaf, grad_outputs=grad_w)[0]
    delta = -float(step_size) * grad_z
    raw_norm = torch.linalg.vector_norm(delta)
    if max_update_norm is not None:
        if max_update_norm <= 0:
            raise ValueError("max_update_norm must be positive or None")
        delta = delta * torch.clamp(
            torch.as_tensor(max_update_norm, dtype=z.dtype, device=z.device)
            / (raw_norm + epsilon), max=1.0
        )
    next_z = (z.detach() + delta).detach()
    return next_z, {
        "latent_gradient_norm": float(torch.linalg.vector_norm(grad_z).detach()),
        "latent_update_norm": float(torch.linalg.vector_norm(delta).detach()),
        "mean_filtered_reference_energy": float(x.square().sum(1).mean().detach()),
    }


@dataclass
class LatentFxLMS:
    """Stateful decoder-backed controller initialized to an exact zero filter."""
    bundle: Any
    z: torch.Tensor
    step_size: float
    epsilon: float = 1e-6
    max_update_norm: float | None = None
    anchor: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.z.ndim != 1:
            raise ValueError("z must be a one-dimensional latent vector")
        self.z = self.z.detach().clone()
        # Remove decoder bias so the initial controller is precisely W = 0.
        with torch.no_grad():
            self.anchor = _decode(self.bundle, self.z).detach()

    @property
    def taps(self) -> torch.Tensor:
        with torch.no_grad():
            return _decode(self.bundle, self.z).detach() - self.anchor

    def update(
        self, filtered_x: torch.Tensor, error: torch.Tensor,
        normalizer: torch.Tensor | None = None,
    ) -> dict[str, float]:
        self.z, diagnostics = latent_fxlms_update(
            self.bundle, self.z, filtered_x, error, normalizer,
            step_size=self.step_size, epsilon=self.epsilon,
            max_update_norm=self.max_update_norm,
        )
        return diagnostics
