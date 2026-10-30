"""Train all six Part-2 autoencoder variants.

Run on the real Part-1 dataset:
    python train.py --data ../W.npy --output outputs/part2

For a clearly-labelled smoke test without Part-1 artefacts:
    python train.py --demo-data --epochs 8 --mixup-samples 32 --output outputs/demo
"""
from __future__ import annotations
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from data import load_filter_dataset, position_split, global_scale
from model import SpectralAutoencoder, FILTER_LENGTH, LATENT_DIM, HIDDEN_DIM
from synthetic import save_demo

MODEL_SPECS = {
    "ae": ("ae", False),
    "ae_mixup": ("ae", True),
    "vae": ("vae", False),
    "vae_mixup": ("vae", True),
    "infovae": ("infovae", False),
    "infovae_mixup": ("infovae", True),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True).T
    return (x2 + y2 - 2.0 * x @ y.T).clamp_min(0.0)


def mmd_rbf(z: torch.Tensor, prior: torch.Tensor, variance: float = 0.01) -> torch.Tensor:
    """Biased mini-batch MMD² with a Gaussian RBF kernel of variance=0.01.

    The very narrow kernel specified in the paper can saturate for 32-D codes;
    this implementation records that value faithfully rather than silently
    changing the kernel. Inspect the logged MMD term when interpreting results.
    """
    if variance <= 0:
        raise ValueError("MMD kernel variance must be positive")
    k_zz = torch.exp(-pairwise_sq_dists(z, z) / (2.0 * variance))
    k_pp = torch.exp(-pairwise_sq_dists(prior, prior) / (2.0 * variance))
    k_zp = torch.exp(-pairwise_sq_dists(z, prior) / (2.0 * variance))
    return k_zz.mean() + k_pp.mean() - 2.0 * k_zp.mean()


def compute_losses(model, x, beta=1.0, mmd_lambda=1000.0,
                   mmd_variance=0.01, sample_vae=True):
    reconstruction, z, mu, logvar = model(x, sample=sample_vae)
    rec_loss = F.mse_loss(reconstruction, x)
    kl_loss = x.new_tensor(0.0)
    mmd_loss = x.new_tensor(0.0)
    if model.kind == "vae":
        # Per-sample KL summed over latent dimensions, averaged over batch.
        kl_loss = -0.5 * torch.sum(1.0 + logvar - mu.square() - logvar.exp(), dim=1).mean()
        total = rec_loss + beta * kl_loss
    elif model.kind == "infovae":
        prior = torch.randn_like(z)
        mmd_loss = mmd_rbf(z, prior, variance=mmd_variance)
        # Standard InfoVAE objective: (1-alpha)*KL + (alpha+lambda-1)*MMD.
        alpha = 1.0
        total = rec_loss + (1.0 - alpha) * kl_loss + (alpha + mmd_lambda - 1.0) * mmd_loss
    else:
        total = rec_loss
    return total, rec_loss, kl_loss, mmd_loss


def validation_reconstruction(model: nn.Module, X: torch.Tensor, batch_size: int) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = X[start:start + batch_size]
            recon, _, _, _ = model(batch, sample=False)
            losses.append(F.mse_loss(recon, batch, reduction="none").mean(dim=1).cpu().numpy())
    return float(np.mean(np.concatenate(losses)))


def mixup_losses(model, X_train: torch.Tensor, n_mixes: int,
                 device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    n = X_train.shape[0]
    ia = torch.randint(0, n, (n_mixes,), device=device)
    ib = torch.randint(0, n, (n_mixes,), device=device)
    w1, w2 = X_train[ia], X_train[ib]
    gamma = torch.rand((n_mixes, 1), device=device)
    w_mix = gamma * w1 + (1.0 - gamma) * w2

    # Consistency reconstruction term L_C.
    rec_mix, _, _, _ = model(w_mix, sample=False)
    loss_c = F.mse_loss(rec_mix, w_mix)

    # Latent linearity term L_z; use mu for probabilistic encoders.
    z1 = model.encode(w1, deterministic=True)
    z2 = model.encode(w2, deterministic=True)
    z_mix = model.encode(w_mix, deterministic=True)
    z_target = gamma * z1 + (1.0 - gamma) * z2
    loss_z = F.mse_loss(z_mix, z_target)
    return loss_c, loss_z


def train_one(name: str, kind: str, use_mixup: bool, X_train: torch.Tensor,
              X_val: torch.Tensor, X_train_all: torch.Tensor, scale: float,
              args, output_dir: Path, device: torch.device) -> dict:
    model = SpectralAutoencoder(kind=kind, latent_dim=args.latent_dim,
                                hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(3, args.patience // 3), min_lr=1e-5
    )
    best_val = float("inf")
    best_state = None
    patience_count = 0
    history = []
    rng = np.random.default_rng(args.seed)
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(X_train.shape[0], device=device)
        totals = {k: [] for k in ["total", "rec", "kl", "mmd", "lc", "lz"]}
        for start in range(0, X_train.shape[0], args.batch_size):
            indices = perm[start:start + args.batch_size]
            x = X_train[indices]
            optimizer.zero_grad(set_to_none=True)
            total, rec, kl, mmd = compute_losses(
                model, x, beta=args.beta, mmd_lambda=args.mmd_lambda,
                mmd_variance=args.mmd_variance, sample_vae=True
            )
            lc = x.new_tensor(0.0)
            lz = x.new_tensor(0.0)
            if use_mixup and args.mixup_samples > 0:
                lc, lz = mixup_losses(model, X_train_all, args.mixup_samples, device)
                total = total + args.mixup_weight_c * lc + args.mixup_weight_z * lz
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite loss in {name} epoch {epoch}")
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            totals["total"].append(float(total.detach().cpu()))
            totals["rec"].append(float(rec.detach().cpu()))
            totals["kl"].append(float(kl.detach().cpu()))
            totals["mmd"].append(float(mmd.detach().cpu()))
            totals["lc"].append(float(lc.detach().cpu()))
            totals["lz"].append(float(lz.detach().cpu()))

        val_rec = validation_reconstruction(model, X_val, args.batch_size)
        scheduler.step(val_rec)
        row = {
            "epoch": epoch,
            "train_total": float(np.mean(totals["total"])),
            "train_reconstruction": float(np.mean(totals["rec"])),
            "train_kl": float(np.mean(totals["kl"])),
            "train_mmd": float(np.mean(totals["mmd"])),
            "train_mixup_reconstruction": float(np.mean(totals["lc"])),
            "train_latent_linearity": float(np.mean(totals["lz"])),
            "validation_reconstruction": val_rec,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(f"[{name:15s}] epoch {epoch:03d}/{args.epochs} "
              f"train_rec={row['train_reconstruction']:.6g} val_rec={val_rec:.6g} "
              f"KL={row['train_kl']:.4g} MMD={row['train_mmd']:.4g}")

        if val_rec < best_val - args.min_delta:
            best_val = val_rec
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
            torch.save({
                "model_state_dict": best_state,
                "global_scale": scale,
                "config": {
                    "name": name, "kind": kind, "use_mixup": use_mixup,
                    "filter_length": FILTER_LENGTH, "latent_dim": args.latent_dim,
                    "hidden_dim": args.hidden_dim, "global_scale": scale,
                    "beta": args.beta, "mmd_lambda": args.mmd_lambda,
                    "mmd_variance": args.mmd_variance,
                    "mixup_samples": args.mixup_samples,
                    "mixup_weight_c": args.mixup_weight_c,
                    "mixup_weight_z": args.mixup_weight_z,
                    "seed": args.seed,
                },
                "best_validation_reconstruction": best_val,
            }, output_dir / "models" / f"{name}.pt")
        else:
            patience_count += 1
        if args.patience > 0 and patience_count >= args.patience:
            print(f"[{name}] early stopping after {epoch} epochs; best val={best_val:.6g}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    with (output_dir / "history" / f"{name}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    return {"name": name, "best_validation_reconstruction": best_val,
            "epochs_run": len(history), "seconds": time.time() - started,
            "checkpoint": str(output_dir / "models" / f"{name}.pt")}


def resolve_data_path(raw: str | None, demo: bool, output_dir: Path) -> Path:
    if demo:
        demo_path = output_dir / "demo_data" / "W_demo_synthetic.npy"
        demo_path.parent.mkdir(parents=True, exist_ok=True)
        if not demo_path.exists():
            save_demo(str(demo_path))
        return demo_path
    if raw:
        return Path(raw)
    candidates = [Path("../W.npy"), Path("W.npy"), Path("../../W.npy")]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find W.npy. Generate it by running the existing Part 1 dataset workflow, "
        "or use --data PATH. For a synthetic smoke test only, pass --demo-data."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None, help="Path to real Part-1 W.npy, shape (2048,512)")
    parser.add_argument("--demo-data", action="store_true", help="Use synthetic demo data (not Part-1 results)")
    parser.add_argument("--output", default="outputs/part2", help="Output folder")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_SPECS) + ["all"], default=["all"])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18, help="Early-stop patience; 0 disables")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mixup-samples", type=int, default=256, help="Convex combinations per batch")
    parser.add_argument("--mixup-weight-c", type=float, default=1.0)
    parser.add_argument("--mixup-weight-z", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0, help="VAE KL weight (paper leaves training details open)")
    parser.add_argument("--mmd-lambda", type=float, default=1000.0)
    parser.add_argument("--mmd-variance", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--val-stride", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-7)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    out = Path(args.output)
    for sub in ["models", "history", "plots", "metrics", "demo_data"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    data_path = resolve_data_path(args.data, args.demo_data, out)
    W = load_filter_dataset(data_path)
    train_idx, val_idx = position_split(W, args.val_stride)
    scale = global_scale(W[train_idx])
    X_scaled = W / scale
    X_train_np, X_val_np = X_scaled[train_idx], X_scaled[val_idx]

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    X_train_all = torch.from_numpy(X_train_np).to(device)
    X_train = X_train_all
    X_val = torch.from_numpy(X_val_np).to(device)

    chosen = list(MODEL_SPECS) if "all" in args.models else args.models
    print(f"Part 2 dataset: {data_path} | shape={W.shape} | synthetic_demo={args.demo_data}")
    print(f"Train={len(train_idx)}, validation={len(val_idx)}, global_scale={scale:.9g}, device={device}")
    metadata = {
        "dataset_path": str(data_path), "dataset_shape": list(W.shape),
        "synthetic_demo": bool(args.demo_data), "train_count": int(len(train_idx)),
        "validation_count": int(len(val_idx)), "validation_stride": int(args.val_stride),
        "global_scale": scale, "train_indices": train_idx.tolist(),
        "validation_indices": val_idx.tolist(), "device": str(device),
        "seed": args.seed, "hyperparameters": vars(args),
        "warning": "All reported metrics are only as real as the supplied W.npy; demo data are synthetic.",
    }
    # Make hyperparameters JSON-safe.
    metadata["hyperparameters"]["models"] = args.models
    with (out / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    np.save(out / "train_indices.npy", train_idx)
    np.save(out / "validation_indices.npy", val_idx)

    results = []
    for name in chosen:
        kind, use_mixup = MODEL_SPECS[name]
        result = train_one(name, kind, use_mixup, X_train, X_val, X_train_all,
                           scale, args, out, device)
        results.append(result)
    with (out / "training_summary.json").open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTraining complete. Checkpoints and logs saved under: {out.resolve()}")
    print("Next: python evaluate.py --data " + str(data_path) + " --output " + str(out))


if __name__ == "__main__":
    main()
