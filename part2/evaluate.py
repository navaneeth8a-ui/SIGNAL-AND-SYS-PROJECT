"""Evaluate trained Part-2 models and save PNG figures + CSV metrics.

Examples:
  python evaluate.py --data ../W.npy --output outputs/part2
  python evaluate.py --data outputs/part2/demo_data/W_demo_synthetic.npy --output outputs/part2

If P.npy and g.npy from Part 1 are available, decoded-filter attenuation is
computed using the same convolution-matrix formula as make_dataset.py.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data import load_filter_dataset, position_split
from model import SpectralAutoencoder, FILTER_LENGTH
from train import MODEL_SPECS

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 180, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def relative_errors(reference: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    num = np.linalg.norm(reconstructed - reference, axis=1)
    den = np.maximum(np.linalg.norm(reference, axis=1), 1e-12)
    return num / den


def attenuation_db(filters: np.ndarray, P_path: Path, g_path: Path) -> np.ndarray:
    """Same formula as the repository's make_dataset.py att_db(W)."""
    P = np.load(P_path)
    g = np.asarray(np.load(g_path)).reshape(-1)
    if P.shape[0] == FILTER_LENGTH:
        pass
    elif P.shape[1] == FILTER_LENGTH:
        P = P.T
    else:
        raise ValueError(f"P must have one dimension of length 512, got {P.shape}")
    if g.size != FILTER_LENGTH:
        raise ValueError(f"g must have length 512, got {g.size}")
    if filters.ndim != 2:
        raise ValueError("filters must have shape (N,512)")
    if filters.shape[1] != FILTER_LENGTH:
        filters = filters.T
    if filters.shape[0] != P.shape[1]:
        raise ValueError(f"Filter count {filters.shape[0]} does not match P columns {P.shape[1]}")
    L = FILTER_LENGTH
    G = np.zeros((2 * L - 1, L), dtype=np.float64)
    for k in range(L):
        G[k:k + L, k] = g
    pp = np.vstack([P, np.zeros((L - 1, P.shape[1]))])
    r = pp + G @ filters.T
    return 10.0 * np.log10((pp ** 2).sum(axis=0) / np.maximum((r ** 2).sum(axis=0), 1e-30))


def numpy_pca_fit(X_train: np.ndarray, n_components: int = 32):
    mean = X_train.mean(axis=0)
    centered = X_train - mean
    # X is 2048x512, so this compact SVD is sufficient and avoids sklearn dependency.
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    total_var = np.sum(singular_values ** 2)
    evr = (singular_values ** 2) / max(total_var, 1e-30)
    return mean, components, evr


def numpy_pca_transform(X: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (X - mean) @ components.T


def numpy_pca_reconstruct(X: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    z = numpy_pca_transform(X, mean, components)
    return z @ components + mean


def load_trained_model(name: str, output: Path, device: torch.device):
    ckpt_path = output / "models" / f"{name}.pt"
    if not ckpt_path.exists():
        return None
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    model = SpectralAutoencoder(kind=cfg["kind"], latent_dim=cfg["latent_dim"],
                                hidden_dim=cfg["hidden_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, float(checkpoint["global_scale"]), checkpoint


def infer(model, W: np.ndarray, scale: float, device: torch.device,
          batch_size: int = 128):
    normalized = W / scale
    Z_parts, decoded_parts = [], []
    with torch.no_grad():
        for start in range(0, len(W), batch_size):
            x = torch.from_numpy(normalized[start:start + batch_size]).to(device)
            z = model.encode(x, deterministic=True)
            decoded = model.decode(z)
            Z_parts.append(z.cpu().numpy())
            decoded_parts.append(decoded.cpu().numpy() * scale)
    return np.concatenate(Z_parts), np.concatenate(decoded_parts)


def blend_metrics(model, W_scaled: np.ndarray, device: torch.device,
                  seed: int = 812, n_pairs: int = 256, batch_size: int = 64):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, len(W_scaled), size=n_pairs)
    b = rng.integers(0, len(W_scaled), size=n_pairs)
    gamma = rng.random((n_pairs, 1)).astype(np.float32)
    w1, w2 = W_scaled[a], W_scaled[b]
    mix = gamma * w1 + (1.0 - gamma) * w2
    recon_parts, z_actual_parts, z_linear_parts = [], [], []
    with torch.no_grad():
        for start in range(0, n_pairs, batch_size):
            sl = slice(start, start + batch_size)
            t1 = torch.from_numpy(w1[sl]).to(device)
            t2 = torch.from_numpy(w2[sl]).to(device)
            tm = torch.from_numpy(mix[sl]).to(device)
            tg = torch.from_numpy(gamma[sl]).to(device)
            z1 = model.encode(t1, deterministic=True)
            z2 = model.encode(t2, deterministic=True)
            zm = model.encode(tm, deterministic=True)
            rec = model.decode(zm)
            zlin = tg * z1 + (1.0 - tg) * z2
            recon_parts.append(rec.cpu().numpy())
            z_actual_parts.append(zm.cpu().numpy())
            z_linear_parts.append(zlin.cpu().numpy())
    rec = np.concatenate(recon_parts)
    z_actual = np.concatenate(z_actual_parts)
    z_linear = np.concatenate(z_linear_parts)
    blend_rel = np.linalg.norm(rec - mix, axis=1) / np.maximum(np.linalg.norm(mix, axis=1), 1e-12)
    z_rel = np.linalg.norm(z_actual - z_linear, axis=1) / np.maximum(np.linalg.norm(z_linear, axis=1), 1e-12)
    latent_mse = float(np.mean((z_actual - z_linear) ** 2))
    return float(np.mean(blend_rel)), float(np.median(blend_rel)), float(np.mean(z_rel)), latent_mse


def jacobian_spectra(model, Z: np.ndarray, device: torch.device, selected=(0.1, 0.5, 0.9), output_scale: float = 1.0):
    outputs = []
    for q in selected:
        index = int(np.clip(round(q * (len(Z) - 1)), 0, len(Z) - 1))
        z = torch.tensor(Z[index], dtype=torch.float32, device=device, requires_grad=True)
        def decode_one(latent):
            return model.decode(latent.unsqueeze(0)).squeeze(0) * output_scale
        jac = torch.autograd.functional.jacobian(decode_one, z, create_graph=False)
        s = torch.linalg.svdvals(jac).detach().cpu().numpy()
        outputs.append((q, s))
    return outputs


def draw_architecture(path: Path):
    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3.2); ax.axis("off")
    boxes = [
        (0.2, 1.0, 1.35, 1.1, "Input filter\n512 taps"),
        (1.8, 1.0, 1.45, 1.1, "RFFT\n257 complex"),
        (3.5, 1.0, 1.5, 1.1, "Real + imag\n514 features"),
        (5.25, 1.0, 1.3, 1.1, "Linear + LN\n+ SiLU · 256"),
        (6.8, 1.0, 1.2, 1.1, "Latent z\n32 values"),
        (8.25, 1.0, 1.3, 1.1, "Linear + LN\n+ SiLU · 256"),
        (9.8, 1.0, 1.1, 1.1, "514 real\nfeatures"),
        (11.1, 1.0, 0.75, 1.1, "IRFFT\n512"),
    ]
    for x, y, w, h, label in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor="#e8f1fa" if "Latent" not in label else "#d6f3e4",
                             edgecolor="#315a7d", linewidth=1.2, joinstyle="round")
        ax.add_patch(rect); ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=8)
    for i in range(len(boxes)-1):
        x, y, w, h, _ = boxes[i]
        xn, yn, wn, hn, _ = boxes[i+1]
        ax.annotate("", xy=(xn-0.03, y+h/2), xytext=(x+w+0.03, y+h/2),
                    arrowprops={"arrowstyle":"->", "lw":1.2, "color":"#315a7d"})
    ax.text(6.9, 2.55, "Encoder: 512 → 514 → 256 → 32", ha="center", fontsize=10, weight="bold")
    ax.text(9.5, 0.45, "Decoder: 32 → 256 → 514 → 512", ha="center", fontsize=10, weight="bold")
    ax.set_title("Part 2 architecture (RFFT / IRFFT are differentiable and have no trainable parameters)", pad=10)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def make_data_plots(W: np.ndarray, output_plots: Path, synthetic: bool):
    n = len(W)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.7))
    positions = np.linspace(0, n-1, min(5, n), dtype=int)
    offsets = np.arange(W.shape[1])
    for i, idx in enumerate(positions):
        axes[0].plot(offsets, W[idx], lw=1, label=f"position {idx}")
    axes[0].set(title="Example filters across source positions", xlabel="Tap index", ylabel="Filter coefficient")
    axes[0].legend(fontsize=7, ncol=1)
    axes[1].plot(np.arange(n), np.max(np.abs(W), axis=1), lw=0.8)
    axes[1].set(title="Per-filter peak magnitude (not used for scaling)", xlabel="Source position index", ylabel="Max |w|")
    suptitle = "Synthetic smoke-test dataset — NOT Part 1 measurements" if synthetic else "Part 1 dataset W.npy"
    fig.suptitle(suptitle, y=1.02, color="darkred" if synthetic else "black")
    fig.tight_layout(); fig.savefig(output_plots / "dataset_overview.png", bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    for idx in positions:
        spec = np.fft.rfft(W[idx], n=512)
        freq = np.fft.rfftfreq(512, d=1/16000)
        db = 20*np.log10(np.maximum(np.abs(spec), 1e-12))
        ax.plot(freq/1000, db, lw=1, label=f"position {idx}")
    ax.set(title="Magnitude spectra of example filters", xlabel="Frequency (kHz)", ylabel="Magnitude (dB)")
    ax.legend(fontsize=7, ncol=2); fig.tight_layout()
    fig.savefig(output_plots / "filter_spectra.png", bbox_inches="tight"); plt.close(fig)


def make_latent_plot(latent_by_model: dict[str, np.ndarray], output: Path, synthetic: bool):
    names = list(latent_by_model)
    if not names:
        return
    ncols = 2; nrows = int(np.ceil(len(names)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.4, 3.25*nrows), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        Z = latent_by_model[name]
        Zc = Z - Z.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(Zc, full_matrices=False)
        coords = Zc @ vt[:2].T
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=np.arange(len(Z)), cmap="viridis", s=5, alpha=0.8)
        ax.set_title(name); ax.set_xlabel("Latent PCA-1"); ax.set_ylabel("Latent PCA-2")
    for ax in axes.ravel()[len(names):]: ax.axis("off")
    fig.suptitle("Latent codes coloured by source position" + (" (synthetic demo)" if synthetic else ""), y=0.99)
    fig.subplots_adjust(left=0.08, right=0.89, bottom=0.06, top=0.93, hspace=0.38, wspace=0.26)
    cax = fig.add_axes([0.92, 0.19, 0.018, 0.62])
    fig.colorbar(sc, cax=cax, label="Source-position index")
    fig.savefig(output, bbox_inches="tight"); plt.close(fig)


def make_summary_plots(metrics: list[dict], histories: dict[str, list[dict]],
                       outputs: dict[str, dict], W: np.ndarray, val_idx: np.ndarray,
                       pca_evr: np.ndarray, pca_metrics: dict, output: Path,
                       synthetic: bool, device: torch.device):
    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, rows in histories.items():
        epochs = [int(r["epoch"]) for r in rows]
        axes[0].plot(epochs, [float(r["train_reconstruction"]) for r in rows], label=name)
        axes[1].plot(epochs, [float(r["validation_reconstruction"]) for r in rows], label=name)
    axes[0].set(title="Training reconstruction MSE", xlabel="Epoch", ylabel="MSE")
    axes[1].set(title="Validation reconstruction MSE", xlabel="Epoch", ylabel="MSE")
    for ax in axes: ax.legend(fontsize=7); ax.set_yscale("log")
    fig.suptitle("Synthetic smoke-test runs" if synthetic else "Training curves")
    fig.tight_layout(); fig.savefig(output / "training_curves.png", bbox_inches="tight"); plt.close(fig)

    # Validation relative error comparison
    names = [m["model"] for m in metrics] + ["PCA-32"]
    means = [m["validation_relative_error_mean"] for m in metrics] + [pca_metrics["validation_relative_error_mean"]]
    medians = [m["validation_relative_error_median"] for m in metrics] + [pca_metrics["validation_relative_error_median"]]
    fig, ax = plt.subplots(figsize=(10.2, 4.3))
    x = np.arange(len(names)); width = 0.38
    ax.bar(x-width/2, means, width, label="Mean")
    ax.bar(x+width/2, medians, width, label="Median")
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.set(title="Relative reconstruction error on held-out positions", ylabel="||ŵ − w|| / ||w||")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "model_comparison.png", bbox_inches="tight"); plt.close(fig)

    # PCA cumulative variance
    cumulative = np.cumsum(pca_evr)
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    upto = min(120, len(cumulative))
    ax.plot(np.arange(1, upto+1), cumulative[:upto], marker=".", ms=3)
    ax.axhline(0.99, ls="--", color="crimson", label="99% variance")
    ax.axvline(32, ls=":", color="grey", label="32 components")
    ax.set(title=f"PCA baseline on training positions ({cumulative[min(31, len(cumulative)-1)]:.1%} by 32 components)",
           xlabel="Number of components", ylabel="Cumulative explained variance", ylim=(0, 1.02))
    ax.legend(); fig.tight_layout(); fig.savefig(output / "pca_cumulative_variance.png", bbox_inches="tight"); plt.close(fig)

    # Reconstruction traces: choose model with smallest validation relative-error mean.
    best_name = min(metrics, key=lambda m: m["validation_relative_error_mean"])["model"] if metrics else None
    if best_name and best_name in outputs:
        original = W[val_idx]
        decoded = outputs[best_name]["decoded"]
        select = np.linspace(0, len(val_idx)-1, min(3, len(val_idx)), dtype=int)
        fig, axes = plt.subplots(len(select), 1, figsize=(10, 2.3*len(select)), squeeze=False)
        for ax, j in zip(axes.ravel(), select):
            idx = int(val_idx[j])
            ax.plot(original[j], lw=1.2, label="Original W")
            ax.plot(decoded[idx], lw=1.0, ls="--", label=f"Decoded ({best_name})")
            ax.set_ylabel("Coefficient"); ax.set_title(f"Held-out source-position index {idx}")
        axes.ravel()[0].legend(fontsize=8)
        fig.suptitle("Held-out filter reconstruction" + (" — synthetic demo" if synthetic else ""), y=1.01)
        fig.tight_layout(); fig.savefig(output / "filter_reconstructions.png", bbox_inches="tight"); plt.close(fig)

    # Blend and latent-linearity plot
    mix_names = [m["model"] for m in metrics]
    if mix_names:
        blend = [m["blend_relative_error_mean"] for m in metrics]
        zlin = [m["latent_linearity_mse"] for m in metrics]
        x = np.arange(len(mix_names)); width = 0.38
        fig, ax = plt.subplots(figsize=(9.7, 4.2))
        ax.bar(x-width/2, blend, width, label="Blend reconstruction relative error")
        ax.bar(x+width/2, zlin, width, label="Latent-linearity MSE (L_z)")
        ax.set_xticks(x, mix_names, rotation=25, ha="right")
        ax.set(title="Mixup stress test on unseen convex combinations", ylabel="Relative error")
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(output / "mixup_consistency.png", bbox_inches="tight"); plt.close(fig)



def stamp_synthetic_disclaimer(plot_dir: Path):
    """Add a visible footer to every demo PNG so it cannot be mistaken for real data."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return
    disclaimer = "SYNTHETIC SMOKE TEST ONLY — NOT PART 1 SIMULATION RESULTS"
    for png in plot_dir.glob("*.png"):
        with Image.open(png).convert("RGB") as image:
            width, height = image.size
            footer = 34
            canvas = Image.new("RGB", (width, height + footer), "white")
            canvas.paste(image, (0, 0))
            draw = ImageDraw.Draw(canvas)
            font = ImageFont.load_default()
            draw.text((width // 2, height + 9), disclaimer, fill=(160, 20, 20), font=font, anchor="mt")
            canvas.save(png)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="W.npy used for training")
    parser.add_argument("--output", default="outputs/part2")
    parser.add_argument("--P", default=None, help="Part-1 P.npy (optional, for exact attenuation metric)")
    parser.add_argument("--g", default=None, help="Part-1 g.npy (optional, for exact attenuation metric)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pairs", type=int, default=256)
    parser.add_argument("--save-decoded", action="store_true", help="Save decoded filters as NPY (large)")
    args = parser.parse_args()
    out = Path(args.output); plots = out / "plots"; metrics_dir = out / "metrics"
    plots.mkdir(parents=True, exist_ok=True); metrics_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    W = load_filter_dataset(args.data)
    metadata_path = out / "metadata.json"
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text())
        train_idx = np.asarray(meta["train_indices"], dtype=int)
        val_idx = np.asarray(meta["validation_indices"], dtype=int)
        synthetic = bool(meta.get("synthetic_demo", False))
    else:
        train_idx, val_idx = position_split(W, 8)
        synthetic = "synthetic" in Path(args.data).name.lower()
    train_scale = float(np.max(np.abs(W[train_idx])))
    make_architecture = plots / "architecture.png"
    draw_architecture(make_architecture)
    make_data_plots(W, plots, synthetic)

    P_path = Path(args.P) if args.P else Path(args.data).resolve().parent / "P.npy"
    g_path = Path(args.g) if args.g else Path(args.data).resolve().parent / "g.npy"
    has_att = P_path.exists() and g_path.exists()
    if not has_att:
        print("P.npy and/or g.npy not found; exact attenuation metrics will be left blank. "
              "Pass --P and --g from Part 1 to enable them.")

    # PCA-32 baseline fit only on training positions.
    X_train_scaled = W[train_idx] / train_scale
    X_val_scaled = W[val_idx] / train_scale
    pca_mean, pca_components, pca_evr = numpy_pca_fit(X_train_scaled, n_components=32)
    pca_train_dec = numpy_pca_reconstruct(X_train_scaled, pca_mean, pca_components) * train_scale
    pca_val_dec = numpy_pca_reconstruct(X_val_scaled, pca_mean, pca_components) * train_scale
    pca_train_err = relative_errors(W[train_idx], pca_train_dec)
    pca_val_err = relative_errors(W[val_idx], pca_val_dec)
    np.savez(out / "pca32_baseline.npz", mean=pca_mean, components=pca_components,
             explained_variance_ratio=pca_evr, global_scale=train_scale)
    pca_metrics = {"model": "PCA-32", "train_relative_error_mean": float(pca_train_err.mean()),
                   "train_relative_error_median": float(np.median(pca_train_err)),
                   "validation_relative_error_mean": float(pca_val_err.mean()),
                   "validation_relative_error_median": float(np.median(pca_val_err)),
                   "pca32_cumulative_variance": float(pca_evr[:32].sum())}

    metrics, histories, latent_by_model, outputs = [], {}, {}, {}
    for name in MODEL_SPECS:
        loaded = load_trained_model(name, out, device)
        if loaded is None:
            print(f"Skipping {name}: checkpoint not found")
            continue
        model, scale, checkpoint = loaded
        Z, decoded_W = infer(model, W, scale, device)
        decoded_scaled = decoded_W / scale
        train_err = relative_errors(W[train_idx], decoded_W[train_idx])
        val_err = relative_errors(W[val_idx], decoded_W[val_idx])
        blend_mean, blend_median, latent_lin, latent_lin_mse = blend_metrics(
            model, (W / scale).astype(np.float32), device, n_pairs=args.pairs
        )
        row = {
            "model": name,
            "best_validation_mse_scaled": float(checkpoint.get("best_validation_reconstruction", np.nan)),
            "train_relative_error_mean": float(train_err.mean()),
            "train_relative_error_median": float(np.median(train_err)),
            "validation_relative_error_mean": float(val_err.mean()),
            "validation_relative_error_median": float(np.median(val_err)),
            "blend_relative_error_mean": blend_mean,
            "blend_relative_error_median": blend_median,
            "latent_linearity_relative_error_mean": latent_lin,
            "latent_linearity_mse": latent_lin_mse,
            "attenuation_median_db": np.nan,
            "attenuation_decoded_min_db": np.nan,
        }
        if has_att:
            try:
                a = attenuation_db(decoded_W, P_path, g_path)
                row["attenuation_median_db"] = float(np.nanmedian(a))
                row["attenuation_decoded_min_db"] = float(np.nanmin(a))
            except Exception as exc:
                warnings.warn(f"Could not compute attenuation for {name}: {exc}")
        metrics.append(row)
        latent_by_model[name] = Z
        outputs[name] = {"decoded": decoded_W, "Z": Z}
        np.save(out / "metrics" / f"Z_{name}.npy", Z.astype(np.float32))
        np.save(out / "metrics" / f"z0_{name}.npy", Z.mean(axis=0).astype(np.float32))
        if args.save_decoded:
            np.save(out / "metrics" / f"W_decoded_{name}.npy", decoded_W.astype(np.float32))
        if has_att:
            np.save(out / "metrics" / f"attenuation_{name}.npy", a.astype(np.float32))
        history_path = out / "history" / f"{name}.csv"
        if history_path.exists():
            with history_path.open() as f:
                histories[name] = list(csv.DictReader(f))
        # Jacobian for every model at three positions.
        js = jacobian_spectra(model, Z, device, output_scale=scale)
        np.savez(out / "metrics" / f"jacobian_singular_values_{name}.npz",
                 **{f"q{q:.1f}": s for q, s in js})

    all_rows = metrics + [pca_metrics]
    if all_rows:
        fields = list(dict.fromkeys(k for r in all_rows for k in r.keys()))
        with (out / "metrics" / "summary_metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader(); writer.writerows(all_rows)
        (out / "metrics" / "summary_metrics.json").write_text(json.dumps(all_rows, indent=2, allow_nan=True))
    primary_name = min(metrics, key=lambda m: m["validation_relative_error_mean"])["model"] if metrics else None
    if primary_name is not None:
        primary_Z = latent_by_model[primary_name]
        np.save(out / "metrics" / "Z.npy", primary_Z.astype(np.float32))
        np.save(out / "metrics" / "z0.npy", primary_Z.mean(axis=0).astype(np.float32))
        (out / "metrics" / "part3_handoff.json").write_text(json.dumps({
            "convenience_Z_model": primary_name,
            "selection_rule": "Lowest held-out relative reconstruction error among available models; re-check decoded-filter attenuation before Part 3.",
            "Z_file": "Z.npy", "z0_file": "z0.npy",
            "note": "Use Z_<model>.npy and z0_<model>.npy to compare any specific model."
        }, indent=2))
    make_latent_plot(latent_by_model, plots / "latent_codes_pca.png", synthetic)
    make_summary_plots(metrics, histories, outputs, W, val_idx, pca_evr, pca_metrics, plots, synthetic, device)

    # Jacobian spectra plot for the model with the lowest validation relative error.
    if metrics:
        best = min(metrics, key=lambda m: m["validation_relative_error_mean"])["model"]
        loaded = load_trained_model(best, out, device)
        model, _, _ = loaded
        _, best_scale, _ = loaded
        js = jacobian_spectra(model, outputs[best]["Z"], device, output_scale=best_scale)
        fig, ax = plt.subplots(figsize=(7.5, 4.0))
        for q, s in js:
            ax.semilogy(np.arange(1, len(s)+1), np.maximum(s, 1e-12), marker=".", ms=4, label=f"position quantile {q:.1f}")
        ax.set(title=f"Decoder Jacobian singular values ({best})", xlabel="Singular-value index", ylabel="Singular value")
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(plots / "jacobian_spectrum.png", bbox_inches="tight"); plt.close(fig)

    print(f"Evaluation complete. PNG figures: {plots.resolve()}")
    print(f"Metrics: {(out / 'metrics' / 'summary_metrics.csv').resolve()}")
    print(f"PCA-32 explained variance: {pca_metrics['pca32_cumulative_variance']:.2%}")
    if synthetic:
        stamp_synthetic_disclaimer(plots)
        print("REMINDER: These are synthetic smoke-test results, not measurements from the Part 1 simulator.")


if __name__ == "__main__":
    main()
