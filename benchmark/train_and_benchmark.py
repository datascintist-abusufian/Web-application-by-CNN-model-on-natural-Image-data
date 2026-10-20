"""
Reproducible 4-model benchmark for the CIFAR-10 Streamlit app.

Trains and evaluates CNN, ResNet50, VGG16 and MobileNetV2 under ONE shared
protocol and writes everything the "Model Comparison" dashboard page needs:

    results/benchmark_summary.json   metadata, learning curves, metrics, latency, robustness
    results/test_predictions.npz     y_true + per-model softmax probabilities on the test split

Protocol (defaults; change with CLI flags):
  * Stratified subsets drawn with a fixed seed:
      train = 1000 img/class (10k), val = 200 img/class (2k, held out from train),
      test  = 500 img/class (5k, from the official CIFAR-10 test set)
  * CNN: the exact architecture of create_cnn_model() in app.py, trained from scratch at 32x32.
  * Transfer models: frozen ImageNet backbone + the exact classifier head used in
    create_transfer_learning_model() in app.py (GAP -> 512 -> 256 -> 10).
    Images are upsampled to --img-size (default 96) and normalised with each
    backbone's own preprocess_input(). Because the backbone is frozen, pooled features
    are extracted once and the head is trained on them (mathematically identical to
    training the full frozen model without augmentation, but ~50x cheaper).
  * Adam(1e-3), batch 64, up to --epochs epochs, early stopping on val_loss
    (patience 8, best weights restored). Full per-epoch curves are kept.
  * Robustness: accuracy on a 1000-image test subset with additive Gaussian noise.

Usage
  python benchmark/train_and_benchmark.py                 # downloads CIFAR-10 via Keras
  python benchmark/train_and_benchmark.py --npz cifar10.npz  # or use a local archive
  (On Colab with a GPU you can raise --img-size to 224 and use the full dataset with
   --train-per-class 4500 --val-per-class 500 --test-per-class 1000.)
"""
import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
from tensorflow.keras import layers, models  # noqa: E402
from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG16  # noqa: E402
from tensorflow.keras.applications import mobilenet_v2, resnet50, vgg16  # noqa: E402

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
NOISE_LEVELS = [0.0, 0.04, 0.08, 0.12, 0.16]   # std of Gaussian noise on [0,1] pixels

BACKBONES = {
    "ResNet50": (ResNet50, resnet50.preprocess_input),
    "VGG16": (VGG16, vgg16.preprocess_input),
    "MobileNetV2": (MobileNetV2, mobilenet_v2.preprocess_input),
}


# --------------------------------------------------------------------------- data
def load_cifar(npz_path=None):
    if npz_path:
        d = np.load(npz_path)
        return (d["x_train"], d["y_train"].ravel(), d["x_test"], d["y_test"].ravel(),
                f"CIFAR-10 from local archive ({os.path.basename(npz_path)})")
    (xtr, ytr), (xte, yte) = tf.keras.datasets.cifar10.load_data()
    return xtr, ytr.ravel(), xte, yte.ravel(), "tf.keras.datasets.cifar10"


def stratified_indices(y, per_class, rng, exclude=None):
    idx = []
    for c in range(10):
        pool = np.where(y == c)[0]
        if exclude is not None:
            pool = np.setdiff1d(pool, exclude)
        idx.append(rng.choice(pool, per_class, replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return idx


def add_noise(x_uint8, sigma, rng):
    x = x_uint8.astype(np.float32) / 255.0
    if sigma > 0:
        x = np.clip(x + rng.normal(0, sigma, x.shape).astype(np.float32), 0, 1)
    return x  # float in [0,1]


# --------------------------------------------------------------------------- models
def create_cnn_model(input_shape=(32, 32, 3), num_classes=10):
    """Identical to create_cnn_model() in app.py."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation="relu", padding="same"), layers.BatchNormalization(),
        layers.Conv2D(32, 3, activation="relu", padding="same"), layers.BatchNormalization(),
        layers.MaxPooling2D(2), layers.Dropout(0.25),
        layers.Conv2D(64, 3, activation="relu", padding="same"), layers.BatchNormalization(),
        layers.Conv2D(64, 3, activation="relu", padding="same"), layers.BatchNormalization(),
        layers.MaxPooling2D(2), layers.Dropout(0.25),
        layers.Conv2D(128, 3, activation="relu", padding="same"), layers.BatchNormalization(),
        layers.Conv2D(128, 3, activation="relu", padding="same"), layers.BatchNormalization(),
        layers.MaxPooling2D(2), layers.Dropout(0.25),
        layers.Flatten(), layers.Dense(256, activation="relu"), layers.BatchNormalization(),
        layers.Dropout(0.5), layers.Dense(num_classes, activation="softmax"),
    ])
    return model


def create_head(feature_dim, num_classes=10):
    """Identical to the classifier head in create_transfer_learning_model() in app.py."""
    return models.Sequential([
        layers.Input(shape=(feature_dim,)),
        layers.Dense(512, activation="relu"), layers.Dropout(0.5),
        layers.Dense(256, activation="relu"), layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])


def compile_and_fit(model, x_tr, y_tr, x_val, y_val, epochs, batch):
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                          restore_best_weights=True)
    t0 = time.time()
    h = model.fit(x_tr, y_tr, validation_data=(x_val, y_val), epochs=epochs,
                  batch_size=batch, callbacks=[es], verbose=2)
    hist = {k: [float(v) for v in vals] for k, vals in h.history.items()}
    best_epoch = int(np.argmin(hist["val_loss"])) + 1
    return hist, best_epoch, time.time() - t0


def to_backbone_input(x01, size, preprocess):
    """x01: float [0,1] (N,32,32,3) -> resized + backbone-specific normalisation."""
    x = tf.image.resize(x01 * 255.0, (size, size), method="bilinear")
    return preprocess(x)


def extract(backbone, x01, size, preprocess, batch=128):
    feats = []
    for i in range(0, len(x01), batch):
        xb = to_backbone_input(x01[i:i + batch], size, preprocess)
        feats.append(backbone(xb, training=False).numpy())
    return np.concatenate(feats)


def time_latency(fn, x_single, x_batch, reps=30):
    fn(x_single); fn(x_batch)  # warm-up / trace
    t = time.perf_counter()
    for _ in range(reps):
        fn(x_single)
    single = (time.perf_counter() - t) / reps * 1000
    t = time.perf_counter()
    for _ in range(max(3, reps // 10)):
        fn(x_batch)
    batched = (time.perf_counter() - t) / max(3, reps // 10) / len(x_batch) * 1000
    return single, batched


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--img-size", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--train-per-class", type=int, default=1000)
    ap.add_argument("--val-per-class", type=int, default=200)
    ap.add_argument("--test-per-class", type=int, default=500)
    ap.add_argument("--robust-per-class", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", default="CNN,ResNet50,VGG16,MobileNetV2")
    ap.add_argument("--save-weights", action="store_true",
                    help="also save the trained CNN and transfer heads to <out>/weights/")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tf.keras.utils.set_random_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    xtr_all, ytr_all, xte_all, yte_all, source = load_cifar(args.npz)
    tr = stratified_indices(ytr_all, args.train_per_class, rng)
    va = stratified_indices(ytr_all, args.val_per_class, rng, exclude=tr)
    te = stratified_indices(yte_all, args.test_per_class, rng)
    x_tr, y_tr = xtr_all[tr].astype(np.float32) / 255., ytr_all[tr]
    x_va, y_va = xtr_all[va].astype(np.float32) / 255., ytr_all[va]
    x_te, y_te = xte_all[te].astype(np.float32) / 255., yte_all[te]

    # robustness subset (first robust-per-class images of each class within test split)
    rob_idx = np.concatenate([np.where(y_te == c)[0][:args.robust_per_class] for c in range(10)])
    noisy_sets = {s: add_noise(xte_all[te][rob_idx], s, np.random.default_rng(args.seed + 1))
                  for s in NOISE_LEVELS}

    summary = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_source": source, "seed": args.seed,
            "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
            "n_robust": int(len(rob_idx)), "noise_levels": NOISE_LEVELS,
            "transfer_input_size": args.img_size, "cnn_input_size": 32,
            "max_epochs": args.epochs, "batch_size": args.batch, "optimizer": "Adam(lr=1e-3)",
            "early_stopping": "val_loss, patience 8, restore best weights",
            "hardware": f"{platform.processor() or platform.machine()} | {os.cpu_count()} CPU cores | "
                        f"GPU: {bool(tf.config.list_physical_devices('GPU'))}",
            "tensorflow": tf.__version__, "class_names": CLASS_NAMES,
        },
        "models": {},
    }
    probs_out = {"y_true": y_te.astype(np.int16)}

    for name in [m.strip() for m in args.models.split(",")]:
        print(f"\n===== {name} =====", flush=True)
        rec = {}
        if name == "CNN":
            model = create_cnn_model()
            hist, best_ep, fit_t = compile_and_fit(model, x_tr, y_tr, x_va, y_va, args.epochs, args.batch)
            predict = lambda x: model(x, training=False).numpy()  # noqa: E731
            probs = np.concatenate([predict(x_te[i:i + 256]) for i in range(0, len(x_te), 256)])
            rob = {s: float((np.concatenate([predict(v[i:i + 256]) for i in range(0, len(v), 256)])
                             .argmax(1) == y_te[rob_idx]).mean()) for s, v in noisy_sets.items()}
            cnn_fn = tf.function(lambda x: model(x, training=False), reduce_retracing=True)
            lat1, latb = time_latency(lambda x: cnn_fn(x).numpy(), x_te[:1], x_te[:64])  # graph mode, like transfer models
            rec.update(total_params=int(model.count_params()), trainable_params=int(model.count_params()),
                       input_size=32, feature_time_s=0.0, pretrained="none (from scratch)")
            if args.save_weights:
                os.makedirs(f"{args.out}/weights", exist_ok=True)
                model.save(f"{args.out}/weights/cnn.keras")
        else:
            cls, prep = BACKBONES[name]
            backbone = cls(weights="imagenet", include_top=False, pooling="avg",
                           input_shape=(args.img_size, args.img_size, 3))
            backbone.trainable = False
            t0 = time.time()
            f_tr = extract(backbone, x_tr, args.img_size, prep)
            f_va = extract(backbone, x_va, args.img_size, prep)
            f_te = extract(backbone, x_te, args.img_size, prep)
            f_rob = {s: extract(backbone, v, args.img_size, prep) for s, v in noisy_sets.items()}
            feat_t = time.time() - t0
            head = create_head(f_tr.shape[1])
            hist, best_ep, fit_t = compile_and_fit(head, f_tr, y_tr, f_va, y_va, args.epochs, args.batch)
            probs = head.predict(f_te, verbose=0, batch_size=512)
            rob = {s: float((head.predict(f, verbose=0).argmax(1) == y_te[rob_idx]).mean())
                   for s, f in f_rob.items()}

            @tf.function(reduce_retracing=True)
            def full(x):
                return head(backbone(to_backbone_input(x, args.img_size, prep), training=False),
                            training=False)
            lat1, latb = time_latency(lambda x: full(x).numpy(), x_te[:1], x_te[:64])
            rec.update(total_params=int(backbone.count_params() + head.count_params()),
                       trainable_params=int(head.count_params()), input_size=args.img_size,
                       feature_time_s=round(feat_t, 1), pretrained="ImageNet (frozen)")
            if args.save_weights:
                os.makedirs(f"{args.out}/weights", exist_ok=True)
                head.save(f"{args.out}/weights/{name.lower()}_head.keras")

        acc = float((probs.argmax(1) == y_te).mean())
        print(f"{name}: test acc {acc:.4f} | best epoch {best_ep} | fit {fit_t:.0f}s | "
              f"latency {lat1:.1f} ms (bs=1) / {latb:.2f} ms/img (bs=64)", flush=True)
        rec.update(history=hist, best_epoch=best_ep, epochs_run=len(hist["loss"]),
                   fit_time_s=round(fit_t, 1), test_accuracy=acc,
                   latency_ms_bs1=round(lat1, 2), latency_ms_per_img_bs64=round(latb, 3),
                   robustness={str(k): v for k, v in rob.items()})
        summary["models"][name] = rec
        probs_out[name] = probs.astype(np.float16)

        # write after every model so partial runs are still usable
        with open(f"{args.out}/benchmark_summary.json", "w") as f:
            json.dump(summary, f, indent=1)
        np.savez_compressed(f"{args.out}/test_predictions.npz", **probs_out)
        tf.keras.backend.clear_session()

    print("\nDone. Results in", args.out)


if __name__ == "__main__":
    main()
