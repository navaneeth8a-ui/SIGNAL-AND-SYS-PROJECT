"""Run matched baseline and latent controllers on the 50-trial Part 1 protocol."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view as swv
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "part2"))
from fxlms import PRE, attenuation_db, block_mse, convergence_time, fxlms
from model import load_bundle
from latent_fxlms import LatentFxLMS

NB, B, SWITCH, L = 400, 100, 100, 512


def run_latent_trial(P, g, ghat, mu, bundle, seed, device):
    rng = np.random.default_rng(seed)
    k1, k2 = rng.choice(P.shape[1], 2, replace=False)
    # Keep the RNG draw and broadband reference identical to Part 1.
    x = rng.standard_normal(PRE + NB * B)
    xhat = lfilter(ghat, 1.0, x)
    r0 = PRE - (L - 1)
    X = swv(x, L)[:, ::-1][r0:r0 + NB * B]
    Xh = swv(xhat, L)[:, ::-1][r0:r0 + NB * B]
    pw = np.convolve(xhat**2, np.ones(L))[PRE:PRE + NB * B] + 1e-6

    ii = np.arange(B)[:, None]
    mm = np.arange(B + L - 1)[None, :]
    j = ii + L - 1 - mm
    Gm = np.where((j >= 0) & (j < L), g[np.clip(j, 0, L - 1)], 0.0)
    Yh = np.zeros((L - 1 + B, 1))
    zero_w = torch.zeros((1, L), dtype=torch.float32, device=device)
    with torch.no_grad():
        z0 = bundle.encode(zero_w)[0]
    controller = LatentFxLMS(bundle, z0, step_size=mu)

    E = np.zeros((NB, B))
    D = np.zeros(NB)
    for block in range(NB):
        sl = slice(block * B, (block + 1) * B)
        path = P[:, k1] if block < SWITCH else P[:, k2]
        w = controller.taps.cpu().numpy().astype(np.float64, copy=False)
        y = X[sl] @ w
        if B >= L - 1:
            Yh[:L - 1] = Yh[B:].copy()
        else:
            Yh[:L - 1] = np.concatenate((Yh[B:L - 1], Yh[L - 1:]))[-(L - 1):]
        Yh[L - 1:, 0] = y
        d = X[sl] @ path
        e = d + (Gm @ Yh)[:, 0]
        E[block] = e
        D[block] = np.mean(d**2)

        # The gradient is exactly the Part 1 block NLMS gradient, then J^T maps
        # it to the decoder's 32-D tangent coordinates.
        controller.update(
            torch.from_numpy(Xh[sl].copy()).float().to(device),
            torch.from_numpy(e.copy()).float().to(device),
            torch.from_numpy(pw[sl].copy()).float().to(device),
        )
    return block_mse(E[:, :, None])[:, 0], D, (int(k1), int(k2))


def summarize(on, off):
    ci = np.array([convergence_time(row, 0.4) for row in on])
    cs = np.array([convergence_time(row, 0.4, start=SWITCH) for row in on])
    att = attenuation_db(off[:, -40:].mean(1), on[:, -40:].mean(1))
    return {
        "conv_initial_mean": float(np.nanmean(ci)),
        "conv_initial_std": float(np.nanstd(ci)),
        "conv_after_switch_mean": float(np.nanmean(cs)),
        "conv_after_switch_std": float(np.nanstd(cs)),
        "atten_db_mean": float(np.nanmean(att)),
        "atten_db_std": float(np.nanstd(att)),
    }


def trial_metrics(on, off):
    ci = np.array([convergence_time(row, 0.4) for row in on])
    cs = np.array([convergence_time(row, 0.4, start=SWITCH) for row in on])
    att = attenuation_db(off[:, -40:].mean(1), on[:, -40:].mean(1))
    clean = lambda values: [None if not np.isfinite(v) else float(v) for v in values]
    return {
        "conv_initial_blocks": clean(ci),
        "conv_after_switch_blocks": clean(cs),
        "attenuation_db": clean(att),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="Trained real-data AE+mixup checkpoint")
    ap.add_argument("--mu", type=float, required=True, help="Latent-space adaptation gain")
    ap.add_argument("--trials", type=int, default=50, choices=range(1, 51))
    ap.add_argument("--output", default="part3/outputs/comparison.json")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))
    device = torch.device(args.device)
    bundle = load_bundle(args.checkpoint, map_location=device).to(device)
    bundle.eval()
    P = np.load(ROOT / "P.npy")
    g = np.load(ROOT / "g.npy")
    ghat = np.load(ROOT / "ghat.npy")

    lat_on, lat_off, positions = [], [], []
    started = time.perf_counter()
    for seed in range(args.trials):
        on, off, pos = run_latent_trial(P, g, ghat, args.mu, bundle, seed, device)
        lat_on.append(on); lat_off.append(off); positions.append(pos)
        print(f"latent mu={args.mu:g} seed={seed + 1}/{args.trials}", flush=True)
    latent_seconds = time.perf_counter() - started
    lat_on, lat_off = np.asarray(lat_on), np.asarray(lat_off)

    base_on, base_off = [], []
    base_started = time.perf_counter()
    for seed in range(args.trials):
        rng = np.random.default_rng(seed)
        rng.choice(P.shape[1], 2, replace=False)  # consume the same path-pair draw
        x = rng.standard_normal(PRE + NB * B)
        W, E, D = fxlms(P[:, [positions[seed][0]]], g, ghat, x, 32,
                        nblocks=NB, P2=P[:, [positions[seed][1]]], switch=SWITCH)
        base_on.append(block_mse(E)[:, 0]); base_off.append(D[:, 0])
    base_on, base_off = np.asarray(base_on), np.asarray(base_off)
    baseline_seconds = time.perf_counter() - base_started
    result = {
        "protocol": "Part 1 broadband Gaussian, seeds 0..N-1; 400 x 100-sample blocks; source path switches at block 100",
        "trials": args.trials,
        "latent_step_size": args.mu,
        "latent": summarize(lat_on, lat_off),
        "baseline_mu_32": summarize(base_on, base_off),
        "latent_per_trial": trial_metrics(lat_on, lat_off),
        "baseline_mu_32_per_trial": trial_metrics(base_on, base_off),
        "latent_wall_seconds": latent_seconds,
        "latent_mean_seconds_per_trial": latent_seconds / args.trials,
        "baseline_wall_seconds": baseline_seconds,
        "positions": positions,
        "latent_attenuation_db": np.round(10 * np.log10(lat_off[:, -40:].mean(1) / lat_on[:, -40:].mean(1)), 4).tolist(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("trials", "latent_step_size", "latent", "baseline_mu_32", "latent_wall_seconds")}, indent=2))


if __name__ == "__main__":
    main()
