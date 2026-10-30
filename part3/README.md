# Part 3 — Latent-space FxLMS

This part adapts the controller in the Part 2 autoencoder's 32-dimensional latent space and compares it against the repository's ordinary 512-tap FxLMS baseline.

## Update rule

For each 100-sample block, the Part 1 filter-space update is

```text
g_w = X_hat.T @ (e / p) / B
p[n] = sum(X_hat[n] ** 2) + epsilon
g_z = J(z).T @ g_w
z = z - mu_z * g_z
```

`latent_fxlms.py` computes the decoder vector-Jacobian product instead of constructing the full 512 × 32 Jacobian. This is mathematically the same projection. The decoder bias is subtracted at initialization so the initial controller is exactly the zero filter. Per-sample power normalization, block averaging, reference pre-roll, path switch, and error sign follow `fxlms.py`.

## Run a matched comparison

Generate Part 1 paths and filters, then train the Part 2 AE+mixup model on the real 2,048-position filter dataset:

```bash
python build_paths.py
python make_dataset.py
python part2/train.py --data W.npy --models ae_mixup --epochs 80 --patience 12 --batch-size 64 --mixup-samples 64 --output part2/outputs/part3-real
```

Then compare latent FxLMS and the μ=32 full-filter baseline with identical seeds and source locations:

```bash
python part3/run_comparison.py --checkpoint part2/outputs/part3-real/models/ae_mixup.pt --mu 80 --trials 50 --output part3/outputs/comparison.json
```

The latent step size is a separate parameter because the decoder Jacobian changes the gradient scale. The reported μz=80 was selected after comparing μz=30, 50, and 80 on this same 50-seed set, so treat it as an exploratory sweep rather than an independent held-out selection. The output JSON records per-trial and aggregate convergence, late-trial attenuation, trial positions, and wall time. For a valid report, use the trained checkpoint and generated `W.npy`; the archived `part2/outputs/demo` checkpoints were trained on synthetic data and are not suitable for the final comparison.

## Baseline reference

Measured on all 50 seeds with the real AE+mixup checkpoint (80 epochs, 64 mixup samples per batch):

| Controller | Recovery after jump (blocks) | Late attenuation (dB) |
|---|---:|---:|
| FxLMS, μ=32 | 73.18 ± 16.24 | 14.69 ± 1.98 |
| Latent FxLMS, μz=80 | 17.30 ± 6.36 | 9.10 ± 1.78 |

The latent controller recovers about 4.2× faster in this run, at the cost of about 5.6 dB in final attenuation. It is a convergence-speed/accuracy tradeoff, not an across-the-board improvement. Under the documented 512-tap room setup, the Wiener attenuation ceiling is about 15.8 dB; the paper's approximately 37 dB result cannot be reproduced from the stated parameters.
