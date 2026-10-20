# Model comparison benchmark

Produces the data behind the **Model Comparison** page (`pages/1_Model_Comparison.py`).

```bash
pip install -r requirements.txt
python benchmark/train_and_benchmark.py      # trains + evaluates CNN, ResNet50, VGG16, MobileNetV2
python benchmark/audit_deployed.py           # evaluates the models exactly as app.py serves them
git add results/benchmark_summary.json results/test_predictions.npz results/deployment_audit.json
```

Outputs (small, safe to commit):

| File | Contents |
|---|---|
| `results/benchmark_summary.json` | protocol metadata, per-epoch learning curves, latency, robustness, params |
| `results/test_predictions.npz` | `y_true` + softmax probabilities per model on the shared test split |
| `results/deployment_audit.json` | accuracy of the models as served by `app.py`, plus findings |

The dashboard computes every metric (accuracy CIs, F1, ECE, ROC/AUC, McNemar, ensemble) from these
files at page load. It does not import TensorFlow, so it opens instantly on Streamlit Cloud.

Default protocol: 10k train / 2k val / 5k test (stratified, seed 42); transfer backbones frozen at
96 px with their own `preprocess_input`; Adam 1e-3, batch 64, ≤40 epochs, early stopping on
val_loss. On a GPU (e.g. Colab) you can scale up:

```bash
python benchmark/train_and_benchmark.py --img-size 224 \
  --train-per-class 4500 --val-per-class 500 --test-per-class 1000
```

`--save-weights` also writes the trained CNN and transfer heads to `results/weights/`, which can be
loaded by `app.py` so the transfer models it serves are actually trained.
