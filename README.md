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
