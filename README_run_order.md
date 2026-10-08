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
