# Run order

Create a virtual environment and install the shared dependencies once. Then run these commands from the repository root:

```bash
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -r requirements.txt
python ism.py
python build_paths.py
python tests.py
python sweep_mu.py
python make_dataset.py
python run_part1.py
python part1/build_replay_data.py --source . --output demo/experiment_data.js
python part2/train.py --data W.npy --models ae_mixup --epochs 80 --patience 12 --batch-size 64 --mixup-samples 64 --output part2/outputs/part3-real
python part2/evaluate.py --data W.npy --output part2/outputs/part3-real --P P.npy --g g.npy
python part3/run_comparison.py --checkpoint part2/outputs/part3-real/models/ae_mixup.pt --mu 80 --trials 50 --output part3/outputs/comparison.json
```

Open `demo/index.html` in a current browser. It loads `demo/experiment_data.js` from the same folder and works offline. The data file can be regenerated after running the Part 1 trials.

The `.npy` path and filter arrays are generated locally and are not tracked in Git. `part2/outputs/demo/` contains synthetic pipeline examples only; use a real `W.npy` and a newly trained checkpoint for Part 3 measurements.
