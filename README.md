# Signals & Systems — broadband active noise control

This repository models room acoustics and compares full-filter FxLMS with a learned latent-space controller.

## Class replay

Open [`demo/index.html`](demo/index.html) with [`experiment_data.js`](demo/experiment_data.js) beside it. It works offline and replays the measured 50-seed broadband experiment: white Gaussian reference, 400 blocks, a source-path jump at block 100, and the selected trial's exact attenuation trace in a 3D room view.

## Main results

| Controller | Recovery after source jump | Late attenuation |
|---|---:|---:|
| Full-filter FxLMS, μ=32 | 73.18 ± 16.24 blocks | 14.69 ± 1.98 dB |
| Latent FxLMS, μz=80 | 17.30 ± 6.36 blocks | 9.10 ± 1.78 dB |

The latent controller recovers about 4.2× faster in this run, at a cost of about 5.6 dB in late attenuation. Both measurements use the same 50 seeds and source jump. The μz sweep used this same seed set and is exploratory, not a held-out hyperparameter study. See [`README_results.md`](README_results.md) and [`part3/README.md`](part3/README.md) for methods and caveats.

## Run the project

Follow [`README_run_order.md`](README_run_order.md) to generate the acoustic paths and filter data, train/evaluate the Part 2 model, rebuild the 50-trial browser data, and run the Part 3 comparison. Arrays and model checkpoints are generated locally and are not tracked in Git; the compact replay data is included.

The documented room's 512-tap Wiener attenuation limit is about 15.8 dB, whereas the paper reports roughly 37 dB. Present the measured results with this setup discrepancy.
