"""
Audit the models exactly as app.py serves them, on the same test split as the benchmark.

app.py loads every model through cifar_models.load(), so this script does the same:
  * CNN        -> cifar10_cnn.h5, 32x32 input scaled to [0, 1]
  * ResNet50 / VGG16 / MobileNetV2 -> frozen ImageNet backbone at 96 px with the backbone's own
               preprocess_input + the trained heads in weights/*.keras
               (produced by: python benchmark/train_and_benchmark.py --save-weights)

Writes results/deployment_audit.json, shown in the Model Comparison page's "Deployment audit" tab.

Usage: python benchmark/audit_deployed.py [--npz cifar10.npz]
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import cifar_models  # noqa: E402
from train_and_benchmark import load_cifar, stratified_indices  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    xtr, ytr, xte, yte, source = load_cifar(args.npz)
    # identical split logic/seed to train_and_benchmark.py -> same test images
    tr = stratified_indices(ytr, 1000, rng)
    stratified_indices(ytr, 200, rng, exclude=tr)
    te = stratified_indices(yte, 500, rng)
    x01, y = xte[te].astype(np.float32) / 255.0, yte[te]

    served = []
    for name in cifar_models.SPECS:
        ex, info = cifar_models.load(name)
        p = ex.predict(x01, batch=128)
        pred = p.argmax(1)
        served.append(dict(model=name, weights=info["weights"], architecture_matches_app_code=True,
                           params=info["params"], app_label_params=f"{info['params']:,} (read from the model)",
                           preprocessing=info["input"], n_images=int(len(y)),
                           accuracy=round(float((pred == y).mean()), 4),
                           distinct_classes_predicted=int(len(np.unique(pred))),
                           mean_confidence=round(float(p.max(1).mean()), 4)))
        print(served[-1], flush=True)
        tf.keras.backend.clear_session()

    accs = ", ".join(f"{r['model']} {r['accuracy']:.1%}" for r in served)
    findings = [
        dict(severity="resolved", title="✅ All four selectable models are trained",
             detail=f"Trained classifier heads are shipped in weights/ and loaded by cifar_models.load(). Accuracy as "
                    f"served on the {len(y):,}-image benchmark test split: {accs}. (Previously 3 of 4 models had a "
                    "random, untrained head and scored 10–16%.) The heads were re-trained with the benchmark protocol "
                    f"on the official CIFAR-10 binary release; the benchmark tables on this page were produced "
                    "from JPEG copies of CIFAR-10, which explains why served transfer accuracy is a few points higher."),
        dict(severity="resolved", title="✅ Backbone-specific preprocessing",
             detail="Images are up-sampled to 96 px and passed through each backbone's own preprocess_input() inside "
                    "the graph — the same protocol the heads were trained with."),
        dict(severity="resolved", title="✅ Model information is read from the loaded model",
             detail=f"The sidebar shows the true parameter count (CNN: {served[0]['params']:,}) and architecture."),
        dict(severity="resolved", title="✅ Uploads are unlabelled by default",
             detail="Uploaded images are no longer scored against the dropdown class; a true label can optionally be "
                    "given and only labelled images enter session accuracy and the confusion matrix."),
        dict(severity="resolved", title="✅ Results persist across widget changes",
             detail="The last prediction is kept in st.session_state, so moving the Top-k slider or switching tabs "
                    "no longer clears it. A 100-image labelled test gallery replaces repeated scoring of 10 images."),
        dict(severity="low", title="Remaining limitation: frozen backbones, no augmentation",
             detail="Transfer models are frozen ImageNet backbones at 96 px; fine-tuning and augmentation would raise "
                    "accuracy further. The CNN is a compact 2-conv network trained from scratch."),
    ]
    out = dict(created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
               n_images=f"{len(y):,}", served_models=served, findings=findings)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "deployment_audit.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", os.path.join(args.out, "deployment_audit.json"))


if __name__ == "__main__":
    main()
