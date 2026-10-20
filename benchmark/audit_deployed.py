"""
Audit the models exactly as app.py serves them, on the same test split as the benchmark.

app.py behaviour reproduced here:
  * CNN  -> downloads cifar10_cnn.h5 from the repo (the only weights file that exists) and
            feeds 32x32 images scaled to [0,1].
  * ResNet50 / VGG16 / MobileNetV2 -> no cifar10_<model>.h5 exists, so app.py builds a NEW
            model (ImageNet backbone + randomly initialised head, never trained) and feeds
            224x224 images scaled to [0,1] (not the backbone's preprocess_input).

Writes results/deployment_audit.json, shown in the dashboard's "Deployment audit" tab.

Usage: python benchmark/audit_deployed.py [--npz cifar10.npz] [--n-transfer 500]
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG16  # noqa: E402
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input  # noqa: E402
from tensorflow.keras.models import Model  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from train_and_benchmark import load_cifar, stratified_indices  # noqa: E402


def app_transfer_model(base_cls):
    """Same construction as create_transfer_learning_model() in app.py."""
    inputs = Input(shape=(224, 224, 3))
    base = base_cls(weights="imagenet", include_top=False, input_tensor=inputs)
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    return Model(inputs, Dense(10, activation="softmax")(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--cnn-weights", default="cifar10_cnn.h5")
    ap.add_argument("--n-transfer", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    xtr, ytr, xte, yte, _ = load_cifar(args.npz)
    # identical split logic/seed to train_and_benchmark.py -> same test images
    tr = stratified_indices(ytr, 1000, rng)
    stratified_indices(ytr, 200, rng, exclude=tr)
    te = stratified_indices(yte, 500, rng)
    x01, y = xte[te].astype(np.float32) / 255.0, yte[te]

    served = []
    cnn = tf.keras.models.load_model(args.cnn_weights, compile=False)
    p = cnn.predict(x01, verbose=0, batch_size=256)
    served.append(dict(model="CNN", weights="cifar10_cnn.h5 (trained)", architecture_matches_app_code=False,
                       params=int(cnn.count_params()), app_label_params="~1.2M",
                       preprocessing="resize 32px, /255", n_images=int(len(y)),
                       accuracy=round(float((p.argmax(1) == y).mean()), 4)))
    print(served[-1], flush=True)

    sub = np.concatenate([np.where(y == c)[0][:args.n_transfer // 10] for c in range(10)])
    for name, cls in (("ResNet50", ResNet50), ("VGG16", VGG16), ("MobileNetV2", MobileNetV2)):
        m = app_transfer_model(cls)
        xs = tf.image.resize(x01[sub], (224, 224)).numpy()
        p = m.predict(xs, verbose=0, batch_size=32)
        pred = p.argmax(1)
        served.append(dict(model=name, weights="none — head created at random on first load",
                           architecture_matches_app_code=True, params=int(m.count_params()),
                           app_label_params={"ResNet50": "~25.6M", "VGG16": "~14.7M", "MobileNetV2": "~3.5M"}[name],
                           preprocessing="resize 224px, /255 (backbone expects preprocess_input)",
                           n_images=int(len(sub)), accuracy=round(float((pred == y[sub]).mean()), 4),
                           distinct_classes_predicted=int(len(np.unique(pred))),
                           mean_confidence=round(float(p.max(1).mean()), 4)))
        print(served[-1], flush=True)
        tf.keras.backend.clear_session()

    bench_path = os.path.join(args.out, "benchmark_summary.json")
    bench = json.load(open(bench_path))["models"] if os.path.exists(bench_path) else {}

    def bacc(mn):
        return f"{bench[mn]['test_accuracy']:.1%}" if mn in bench else "n/a"

    cnn_acc = served[0]["accuracy"]
    tr_accs = ", ".join(f"{r['model']} {r['accuracy']:.1%}" for r in served[1:])
    findings = [
        dict(severity="high", title="3 of the 4 selectable models are untrained",
             detail=f"No cifar10_resnet50/vgg16/mobilenetv2.h5 exists in the repo, so app.py builds these models "
                    f"with a randomly initialised classifier head and never trains it. Measured accuracy as served: "
                    f"{tr_accs} — near chance (10%), collapsing onto only "
                    f"{min(r['distinct_classes_predicted'] for r in served[1:])}–"
                    f"{max(r['distinct_classes_predicted'] for r in served[1:])} distinct classes, and a fresh random "
                    "head is drawn every time the app restarts. The same architectures trained in this benchmark reach "
                    f"ResNet50 {bacc('ResNet50')}, VGG16 {bacc('VGG16')}, MobileNetV2 {bacc('MobileNetV2')}. "
                    "Fix: ship trained heads (benchmark --save-weights) and load them in load_or_create_model()."),
        dict(severity="high", title="Wrong input normalisation for ImageNet backbones",
             detail="preprocess_image() divides every image by 255. ResNet50/VGG16 expect caffe-style mean "
                    "subtraction (0–255, BGR) and MobileNetV2 expects [-1, 1]. Use each backbone's "
                    "preprocess_input(), as the benchmark does."),
        dict(severity="medium", title="Served CNN ≠ CNN described in the code",
             detail=f"cifar10_cnn.h5 is a 2-conv/2-dense network with {served[0]['params']:,} parameters, not the "
                    "6-conv BatchNorm model in create_cnn_model() (~0.82M) and not the '~1.2M' shown in the sidebar. "
                    f"Its accuracy on the benchmark test split is {cnn_acc:.1%} vs {bacc('CNN')} for the "
                    "benchmark CNN."),
        dict(severity="medium", title="Uploaded images are scored against the dropdown class",
             detail="When a user uploads an image, 'Correct/Incorrect', the session accuracy and the confusion "
                    "matrix still use the selected dropdown class as ground truth, so these metrics are "
                    "contaminated. Mark uploads as 'unlabelled' or ask for the true label."),
        dict(severity="low", title="Session 'Research Metrics' rest on 10 fixed images",
             detail="Only one sample image per class is available, so the sidebar accuracy re-scores the same "
                    "images repeatedly. Use this dashboard's held-out test metrics (with CIs) for reporting."),
        dict(severity="low", title="Results disappear when the Top-k slider moves",
             detail="All results render inside `if st.button(...)`; any widget change reruns the script with the "
                    "button False. Store the last prediction in st.session_state and render from there."),
    ]
    out = dict(created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
               n_images=f"{len(y):,} (CNN) / {len(sub):,} (transfer)", served_models=served, findings=findings)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "deployment_audit.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", os.path.join(args.out, "deployment_audit.json"))


if __name__ == "__main__":
    main()
