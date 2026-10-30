"""Spectral autoencoders used in Part 2 of the Latent FxLMS project.

Input/output filters have 512 taps. RFFT produces 257 complex values; their
real and imaginary parts are concatenated into 514 real features. The inverse
transform returns a real, 512-tap filter. All FFT operations remain differentiable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

FILTER_LENGTH = 512
SPECTRAL_FEATURES = 514
LATENT_DIM = 32
HIDDEN_DIM = 256


class SpectralAutoencoder(nn.Module):
    """Shared architecture for AE, VAE, and InfoVAE variants.

    kind: "ae", "vae", or "infovae". Mixup is a training objective, not a
    separate architecture; use the same model with mixup losses enabled.
    """
    def __init__(self, kind: str = "ae", latent_dim: int = LATENT_DIM,
                 hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        if kind not in {"ae", "vae", "infovae"}:
            raise ValueError("kind must be 'ae', 'vae', or 'infovae'")
        self.kind = kind
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)

        self.enc_fc1 = nn.Linear(SPECTRAL_FEATURES, hidden_dim)
        self.enc_norm = nn.LayerNorm(hidden_dim)
        self.enc_fc2 = nn.Linear(hidden_dim, latent_dim * (2 if kind in {"vae", "infovae"} else 1))

        self.dec_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_norm = nn.LayerNorm(hidden_dim)
        self.dec_fc2 = nn.Linear(hidden_dim, SPECTRAL_FEATURES)

    @staticmethod
    def _to_spectrum(w: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(w, n=FILTER_LENGTH, dim=-1)
        return torch.cat((spectrum.real, spectrum.imag), dim=-1)

    @staticmethod
    def _from_spectrum(features: torch.Tensor) -> torch.Tensor:
        real, imag = features[..., :257], features[..., 257:]
        spectrum = torch.complex(real, imag)
        return torch.fft.irfft(spectrum, n=FILTER_LENGTH, dim=-1)

    def _encode_features(self, w: torch.Tensor) -> torch.Tensor:
        if w.shape[-1] != FILTER_LENGTH:
            raise ValueError(f"Expected filters of length {FILTER_LENGTH}, got {w.shape[-1]}")
        features = self._to_spectrum(w)
        hidden = F.silu(self.enc_norm(self.enc_fc1(features)))
        return self.enc_fc2(hidden)

    def encode_stats(self, w: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return deterministic code for AE or (mu, logvar) for VAE/InfoVAE."""
        result = self._encode_features(w)
        if self.kind == "ae":
            return result, None
        mu, logvar = result.chunk(2, dim=-1)
        # Clamping avoids overflow in exp(logvar) during early training.
        return mu, logvar.clamp(min=-12.0, max=8.0)

    def encode(self, w: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Encode original normalized filters to latent codes.

        For probabilistic models, deterministic=True returns mu. This is used
        for evaluation and for the latent-linearity mixup penalty.
        """
        first, second = self.encode_stats(w)
        if self.kind == "ae" or deterministic:
            return first
        std = torch.exp(0.5 * second)
        return first + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected latent dimension {self.latent_dim}, got {z.shape[-1]}")
        hidden = F.silu(self.dec_norm(self.dec_fc1(z)))
        spectral_features = self.dec_fc2(hidden)
        return self._from_spectrum(spectral_features)

    def forward(self, w: torch.Tensor, sample: Optional[bool] = None):
        mu, logvar = self.encode_stats(w)
        if self.kind == "ae":
            z = mu
        else:
            if sample is None:
                sample = self.training
            if sample:
                z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            else:
                z = mu
        reconstruction = self.decode(z)
        return reconstruction, z, mu, logvar


class OriginalUnitsBundle(nn.Module):
    """Adapter for Part 3: encode/decode in original filter units.

    The learned network itself is trained on globally scaled filters. Store the
    single global scale in each checkpoint; this wrapper restores original units.
    """
    def __init__(self, network: SpectralAutoencoder, global_scale: float):
        super().__init__()
        self.network = network
        self.register_buffer("global_scale", torch.tensor(float(global_scale)))

    def encode(self, w_original: torch.Tensor) -> torch.Tensor:
        return self.network.encode(w_original / self.global_scale, deterministic=True)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.network.decode(z) * self.global_scale

    def forward(self, w_original: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(w_original))


def load_bundle(checkpoint_path: str, map_location: str | torch.device = "cpu") -> OriginalUnitsBundle:
    """Load a trained checkpoint as an encode/decode adapter in original units."""
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    cfg = checkpoint["config"]
    model = SpectralAutoencoder(
        kind=cfg["kind"], latent_dim=cfg["latent_dim"], hidden_dim=cfg["hidden_dim"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    bundle = OriginalUnitsBundle(model, checkpoint["global_scale"])
    bundle.eval()
    return bundle
