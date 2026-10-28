# Part 1 results: acoustic simulation and baseline FxLMS

## Setup (as in the paper)
Room 6 x 6.2 x 3 m, RT60 0.15 s (Sabine absorption 0.812), fs = 16 kHz, 512-tap impulse responses.
Secondary speaker [3, 2.5, 1.5], error mic [4.5, 3, 1.5], 2,048 primary positions on the line [1.5,1,1] to [3,2,2]
(about 1.0 mm apart; the paper says roughly 2 mm). Primary source is always farther from the mic than the speaker (1.87 m min vs 1.58 m).
Reference noise: white Gaussian. g-hat = g. Block size 100 samples (6.25 ms), normalized update (Eq. 4) averaged over the block.

## Validation
- `tests.py`: vectorised FxLMS equals a sample-by-sample reference to 2e-16.
- Converged FxLMS filters (mu = 16, 3000 blocks) match the closed-form Wiener optimum: median relative distance 3.2%, attenuation 15.7 vs 15.8 dB.

## Dataset
`W.npy` (2048 x 512, float32) = converged FxLMS filters. The main arrival tap slides 82 taps along the path.
Global linear PCA needs 81 components for 99% of the variance (32 components: 61%), so the manifold is curved.
Locally (64 neighbouring positions) 3 components give 99%. Neighbouring filters are about 20x closer than random pairs.

## Baseline FxLMS, 50 trials, 400 blocks, source jumps at block 100 (convergence rho = 0.4, e_inf from last 40 blocks)
| step size mu | conv. time from start (blocks) | conv. time after jump (blocks) | steady-state attenuation (dB) |
|---|---|---|---|
| 8 | 279 ± 31 | 182 ± 17 | 13.6 ± 0.6 |
| 16 | 195 ± 69 | 136 ± 15 | 15.5 ± 0.8 |
| 24 | 115 ± 62 | 100 ± 13 | 15.5 ± 0.8 |
| 32 | 70 ± 37 | 73 ± 16 | 14.7 ± 2.0 |

Step size 48 diverges. mu = 32 is the largest well-behaved step and is the baseline to beat in Part 3.

## Caveat: attenuation ceiling
The paper reports about 37 dB. With these stated parameters the best possible attenuation for a full-band white reference is about 15.8 dB
(median Wiener optimum; range 13.7 to 18.2 dB): truncating the responses at 512 taps leaves energy that a 512-tap filter cannot cancel (the
speaker path adds a 74-sample delay). Bandlimiting the noise or tapering the responses did not change this. The paper's setup must differ
in an unreported detail. Relative comparisons (convergence speed) remain valid.

# Running Part 1 (needs only numpy, scipy, matplotlib)

    python3 ism.py            # sanity check of the room simulation
    python3 build_paths.py    # impulse responses -> P.npy, g.npy, ghat.npy, positions.npy, W_wiener.npy
    python3 tests.py          # validates the fast FxLMS against a plain reference (prints PASS)
    python3 sweep_mu.py       # step-size sweep (finds the stability limit)
    python3 make_dataset.py   # converged FxLMS filters -> W.npy (about 90 s)
    python3 run_part1.py      # 50-trial experiments -> curves.npy, results.json
    python3 make_figures.py   # figures/ and pca.json

Files: ism.py (room), fxlms.py (block FxLMS + metrics + Wiener optimum), trials.py (trial harness; Part 3 plugs in here).
Shipped outputs (P.npy, W.npy, etc.) are already included, so Part 2 and Part 3 can start without rerunning anything.

# Part 2 — Latent FxLMS autoencoders

This folder implements the autoencoder stage for the Signals & Systems project and is designed to sit in the root of `navaneeth8a-ui/SIGNAL-AND-SYS-PROJECT` as a new `part2/` folder. It consumes the existing Part 1 `W.npy` output (shape **2048 × 512**) and writes trained model checkpoints, latent codes, evaluation tables, and PNG figures for Part 3.

## 1. What is included

- `model.py` — shared differentiable spectral encoder/decoder and original-units wrapper for Part 3.
- `train.py` — trains all six variants: AE, AE+mixup, VAE, VAE+mixup, InfoVAE, InfoVAE+mixup.
- `evaluate.py` — position-held-out evaluation, PCA-32 baseline, blend test, Jacobian singular values, attenuation if `P.npy` and `g.npy` are available, and report-quality PNGs.
- `data.py` — strict `W.npy` loading, position-aware split, one global scale (not per-filter scaling).
- `synthetic.py` — synthetic dataset generator for a pipeline smoke test only. It is **not** a substitute for Part 1 measurements.
- `smoke_test.py` — checks shapes, FFT backpropagation, losses, and the train/validation split.
- `outputs/demo/` (created when you run the demo) — example CSV/JSON outputs and PNGs clearly labelled synthetic.

## 2. Install and use with the real Part 1 dataset

From the repository root, after the existing Part 1 workflow has generated `W.npy`:

```bash
python3 -m venv .venv
source .venv/bin/activate                 # macOS/Linux
python -m pip install --upgrade pip
python -m pip install -r part2/requirements.txt
python part2/smoke_test.py
python part2/train.py --data W.npy --output part2/outputs/part2 --epochs 120
python part2/evaluate.py --data W.npy --output part2/outputs/part2 --P P.npy --g g.npy
```

If `P.npy` and `g.npy` are stored in a different directory, pass their actual paths to `--P` and `--g`. If those arrays are unavailable, evaluation still produces reconstruction, PCA, blend, latent, and Jacobian figures; it leaves the exact attenuation fields blank rather than fabricating them.

`W.npy` is generated by the existing Part 1 workflow; it is not part of the tracked source listing. The Part 1 repository notes state that it has shape `(2048, 512)` and that the dataset stores converged FxLMS filters. The scale factor used during training is saved in every checkpoint and in `outputs/part2/metadata.json`.

## 3. Quick demo without Part 1 artefacts

This exercises the end-to-end code path using a reproducible synthetic dataset. Its metrics must **not** be described as real simulation results or as a reproduction of the paper.

```bash
python part2/train.py --demo-data --epochs 8 --patience 0 --batch-size 128 --mixup-samples 32 --output part2/outputs/demo
python part2/evaluate.py --data part2/outputs/demo/demo_data/W_demo_synthetic.npy --output part2/outputs/demo
```

The demo command intentionally uses fewer epochs and fewer mixup combinations than the research run to make a fast smoke test. For a proper run, use the real `W.npy`, batch size 64, and 256 mixup combinations per batch.

## 4. Architecture

For each input filter `w` (512 real taps):

1. `torch.fft.rfft(w, n=512)` → 257 complex coefficients.
2. Concatenate real and imaginary parts → 514 real features.
3. `Linear(514,256) → LayerNorm(256) → SiLU → Linear(256,32)`.
4. Decoder: `Linear(32,256) → LayerNorm(256) → SiLU → Linear(256,514)`.
5. Split into real/imaginary parts and run `torch.fft.irfft(..., n=512)` → 512-tap reconstructed filter.

The FFT operations have no learned parameters but stay in the autograd graph. `SiLU(x) = x sigmoid(x)` is smooth, which is useful for the decoder Jacobian needed by Part 3.

## 5. Six models and objective

- **AE:** reconstruction MSE.
- **AE+mixup:** reconstruction MSE plus `L_C` (mixed-filter reconstruction consistency) and `L_z` (latent linearity).
- **VAE:** reconstruction MSE plus Gaussian KL divergence, using `z = μ + σ ε`.
- **VAE+mixup:** VAE objective plus `L_C + L_z`.
- **InfoVAE:** reconstruction plus MMD regularization with Gaussian kernel variance `0.01`, `α=1`, `λ=1000` as stated in the project notes.
- **InfoVAE+mixup:** InfoVAE objective plus `L_C + L_z`.

The paper does not specify all training choices, including optimizer/epoch count, train-validation split, or the precise MSE/KL reduction. This implementation documents them in `metadata.json`. Defaults are Adam at `1e-3`, position holdout every eighth sample, early stopping, a single global max-absolute scale, VAE `beta=1`, and mixup weights 1.0.

**InfoVAE kernel caution:** a Gaussian variance of 0.01 is narrow in a 32-dimensional latent space, so the MMD can saturate (most pairwise kernel values close to zero). We preserve the stated value by default; inspect `train_mmd` in the CSV and consider a documented kernel sensitivity study if the regularizer appears ineffective. Do not silently change the paper's value and claim exact reproduction.

## 6. Generated outputs

After training and evaluating, inspect `outputs/part2/plots/`:

- `architecture.png`
- `dataset_overview.png`, `filter_spectra.png`
- `training_curves.png`
- `model_comparison.png`
- `filter_reconstructions.png`
- `mixup_consistency.png`
- `latent_codes_pca.png`
- `pca_cumulative_variance.png`
- `jacobian_spectrum.png`

Metrics and code artifacts go to `outputs/part2/metrics/`: `summary_metrics.csv/json`, `Z_<model>.npy`, `z0_<model>.npy`, Jacobian singular values, and optional decoded filters. `Z.npy` / `z0.npy` are convenience copies of the first trained model; prefer the explicitly named `Z_<model>.npy` and `z0_<model>.npy` for Part 3.

## 7. Exact attenuation metric

When `P.npy` and `g.npy` are supplied, `evaluate.py` uses the same operation as Part 1's `make_dataset.py`: build the secondary-path convolution matrix `G`, pad `P` with 511 zeros, form `r = p_padded + G @ W.T`, and calculate `10 log10(sum(p_padded²)/sum(r²))`. This is attenuation for the *decoded filters themselves*, not just reconstruction MSE. Note that the Part 1 README reports a median Wiener-optimum attenuation near 15.8 dB under this simulation setup, while the paper reports about 37 dB; keep that caveat in any report until the setup discrepancy is resolved.

## 8. Part 3 integration

```python
import torch
from model import load_bundle
bundle = load_bundle("part2/outputs/part2/models/ae_mixup.pt")

# Original filter units in, original filter units out:
w = torch.zeros(1, 512)
z = bundle.encode(w)         # (1, 32)
w_hat = bundle.decode(z)     # (1, 512), original units

# Decoder Jacobian (512 x 32) for latent FxLMS:
z0 = z[0].detach().requires_grad_(True)
J = torch.autograd.functional.jacobian(
    lambda zz: bundle.decode(zz.unsqueeze(0)).squeeze(0), z0
)
```

Choose a model based on held-out reconstruction **and** decoded-filter attenuation, not only its training loss. For the final Part 3 baseline, the project notes specify `μ=32` FxLMS convergence at around 70 blocks from the start and 73 blocks after the source jump under the current Part 1 setup.
