"""Fast checks for network shapes, FFT differentiability, losses, and the data split."""
from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np
from model import SpectralAutoencoder
from train import compute_losses, mixup_losses
from data import position_split


def main():
    torch.set_num_threads(2)
    x = torch.randn(8, 512, requires_grad=False)
    for kind in ("ae", "vae", "infovae"):
        model = SpectralAutoencoder(kind=kind)
        recon, z, mu, logvar = model(x, sample=True)
        assert recon.shape == (8, 512), (kind, recon.shape)
        assert z.shape == (8, 32), (kind, z.shape)
        loss, rec, kl, mmd = compute_losses(model, x, sample_vae=True)
        assert torch.isfinite(loss)
        loss.backward()
        assert model.enc_fc1.weight.grad is not None
        assert model.dec_fc2.weight.grad is not None
        print(f"{kind}: recon={tuple(recon.shape)}, z={tuple(z.shape)}, grad=OK")
    model = SpectralAutoencoder(kind="ae")
    lc, lz = mixup_losses(model, x, n_mixes=12, device=torch.device("cpu"))
    assert torch.isfinite(lc) and torch.isfinite(lz)
    W = np.arange(20 * 512, dtype=np.float32).reshape(20, 512)
    tr, va = position_split(W, 4)
    assert len(set(tr).intersection(set(va))) == 0
    assert np.all(va % 4 == 0)
    print(f"mixup: L_C={lc.item():.4g}, L_z={lz.item():.4g}; split: {len(tr)} train/{len(va)} validation")
    print("All smoke tests passed.")

if __name__ == "__main__": main()
