"""
Train and audit CNNs on two MedMNIST v2 (MedMNIST+ 64 px) tasks and pre-compute every
analysis shown on the "Medical Imaging" Streamlit page.

    PneumoniaMNIST  paediatric chest X-ray, binary (normal vs pneumonia), grey-scale
    BloodMNIST      peripheral blood-cell microscopy, 8 classes, RGB

Outputs  results/medical/<dataset>/
    model.h5          Keras model (TF 2.15) used live on the page for XAI / MC-Dropout
    analysis.npz      per-image arrays: probabilities, MC-Dropout, embeddings, OOD scores, ...
    summary.json      metadata, training curves, temperature, robustness, faithfulness

Usage
    python benchmark/train_medical.py --data-dir data/          # expects <name>_64.npz
    (download: https://zenodo.org/records/10519652/files/pneumoniamnist_64.npz
               https://zenodo.org/records/10519652/files/bloodmnist_64.npz
     and the CIFAR-10 test set is used as a far-OOD set: --cifar-npz cifar10.npz)

Reference: Yang J. et al. MedMNIST v2. Scientific Data 10, 41 (2023).
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
from tensorflow.keras import layers, models  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import xai_utils as X  # noqa: E402

DATASETS = {
    "pneumoniamnist": dict(
        title="PneumoniaMNIST", modality="Paediatric chest X-ray (anterior-posterior)",
        classes=["normal", "pneumonia"], channels=1, positive=1, epochs=40,
        source="Kermany et al., Cell 2018 (Guangzhou Women and Children's Medical Center), via MedMNIST v2"),
    "bloodmnist": dict(
        title="BloodMNIST", modality="Peripheral blood smear microscopy (CellaVision DM96)",
        classes=["basophil", "eosinophil", "erythroblast", "immature granulocyte",
                 "lymphocyte", "monocyte", "neutrophil", "platelet"],
        channels=3, positive=None, epochs=25,
        source="Acevedo et al., Data in Brief 2020 (Hospital Clinic de Barcelona), via MedMNIST v2"),
}
FGSM_EPS = [0, 0.5, 1, 2, 4, 8]          # in /255 units
SEVERITIES = [1, 2, 3, 4, 5]


def build_model(input_shape, n_classes):
    return models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, padding="same", activation="relu", name="conv1"), layers.BatchNormalization(momentum=0.9),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, padding="same", activation="relu", name="conv2"), layers.BatchNormalization(momentum=0.9),
        layers.MaxPooling2D(2), layers.SpatialDropout2D(0.1),
        layers.Conv2D(128, 3, padding="same", activation="relu", name="conv3"), layers.BatchNormalization(momentum=0.9),
        layers.Conv2D(128, 3, padding="same", activation="relu", name="last_conv"), layers.BatchNormalization(momentum=0.9),
        layers.GlobalAveragePooling2D(name="gap"),
        layers.Dense(128, activation="relu", name="embedding"),
        layers.Dropout(0.4, name="mc_dropout"),
        layers.Dense(n_classes, activation="softmax", name="probs"),
    ], name="medcnn")


def augment_fn(rgb):
    def aug(x, y):
        x = tf.image.random_brightness(x, 0.08)
        x = tf.image.random_contrast(x, 0.85, 1.15)
        if rgb:  # cells are rotation/flip invariant; chest X-rays are not flipped
            x = tf.image.random_flip_left_right(x)
            x = tf.image.random_flip_up_down(x)
            x = tf.image.rot90(x, tf.random.uniform([], 0, 4, tf.int32))
        pad = tf.image.pad_to_bounding_box(x, 4, 4, 72, 72)
        x = tf.image.random_crop(pad, tf.shape(x))
        return tf.clip_by_value(x, 0, 1), y
    return aug


def embed(model, x, batch=256):
    layers_ = model.layers[: [l.name for l in model.layers].index("embedding") + 1]
    out = []
    for i in range(0, len(x), batch):
        out.append(X._run_layers(layers_, tf.convert_to_tensor(x[i:i + batch], tf.float32)).numpy())
    return np.concatenate(out)


def to_channels(x01, ch):
    if ch == 1 and x01.shape[-1] == 3:
        return X.to_gray(x01)[..., None].astype(np.float32)
    if ch == 3 and x01.shape[-1] == 1:
        return np.repeat(x01, 3, -1)
    return x01


def macro_auc(y, p):
    from sklearn.metrics import roc_auc_score
    if p.shape[1] == 2:
        return float(roc_auc_score(y, p[:, 1]))
    return float(roc_auc_score(y, p, multi_class="ovr", average="macro"))


def logits(model, x, batch=256):
    """Pre-softmax logits: embedding → final Dense kernel (energy scores need true logits)."""
    W, b = model.get_layer("probs").get_weights()
    return embed(model, x, batch) @ W + b


def compute_ood(model, ex, name, args, cifar_test, xtr, ytr, xte, ch, C):
    e_tr = embed(model, xtr)
    mus, prec = X.fit_mahalanobis(e_tr, ytr, C)
    other = "bloodmnist" if name == "pneumoniamnist" else "pneumoniamnist"
    od = np.load(os.path.join(args.data_dir, f"{other}_64.npz"))["test_images"][:500].astype(np.float32) / 255
    od = od[..., None] if od.ndim == 3 else od
    n_ood = 500
    noise_rng = np.random.default_rng(args.seed + 7)
    ood_sets = {
        "CIFAR-10 natural images (far-OOD)": to_channels(
            tf.image.resize(cifar_test[:n_ood].astype(np.float32) / 255, (64, 64)).numpy(), ch),
        f"{DATASETS[other]['title']} (other modality)": to_channels(od, ch),
        "Uniform noise": noise_rng.random((n_ood,) + xte.shape[1:]).astype(np.float32),
        "Intensity-inverted test images (near-OOD)": 1 - xte[:n_ood],
    }

    def scores(x):
        p = ex.predict(x)
        s = X.mc_dropout(ex, x, T=15)
        _, _, _, epi = X.decompose(s)
        return {"MSP": p.max(1), "Energy": X.energy_score(logits(model, x)), "Mahalanobis":
                X.mahalanobis_score(embed(model, x), mus, prec), "MC epistemic (−MI)": -epi}
    id_scores = scores(xte)
    ood_scores = {k: scores(v) for k, v in ood_sets.items()}
    ood_table = {k: {m: X.ood_metrics(id_scores[m], v[m]) for m in v} for k, v in ood_scores.items()}
    print("OOD AUROC:", {k: {m: round(a[0], 3) for m, a in v.items()} for k, v in ood_table.items()}, flush=True)
    return mus, prec, ood_table, id_scores, ood_scores


def ood_only(name, args, cifar_test):
    """Recompute only the OOD analysis for an already-trained model (no retraining)."""
    cfg = DATASETS[name]
    out_dir = os.path.join(args.out, name)
    d = np.load(os.path.join(args.data_dir, f"{name}_64.npz"))
    prep = lambda a: (a.astype(np.float32) / 255.0)[..., None] if a.ndim == 3 else a.astype(np.float32) / 255.0  # noqa
    xtr, xte = prep(d["train_images"]), prep(d["test_images"])
    ytr = d["train_labels"].ravel().astype(int)
    model = tf.keras.models.load_model(os.path.join(out_dir, "model.h5"), compile=False)
    ex = X.explainable_from_sequential(model, "last_conv", name=cfg["title"])
    mus, prec, ood_table, id_scores, ood_scores = compute_ood(
        model, ex, name, args, cifar_test, xtr, ytr, xte, cfg["channels"], len(cfg["classes"]))
    arr = dict(np.load(os.path.join(out_dir, "analysis.npz")))
    arr = {k: v for k, v in arr.items() if not k.startswith("ood")}
    arr.update(maha_mus=mus.astype(np.float32), maha_prec=prec.astype(np.float32),
               **{f"ood_id__{m}": v.astype(np.float32) for m, v in id_scores.items()},
               **{f"ood__{k}__{m}": v.astype(np.float32) for k, s in ood_scores.items() for m, v in s.items()})
    np.savez_compressed(os.path.join(out_dir, "analysis.npz"), **arr)
    summ = json.load(open(os.path.join(out_dir, "summary.json")))
    summ["ood"] = ood_table
    json.dump(summ, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    print("OOD updated →", out_dir, flush=True)


def run(name, args, cifar_test):
    cfg = DATASETS[name]
    print(f"\n========== {cfg['title']} ==========", flush=True)
    out_dir = os.path.join(args.out, name)
    os.makedirs(out_dir, exist_ok=True)
    d = np.load(os.path.join(args.data_dir, f"{name}_64.npz"))

    def prep(a):
        a = a.astype(np.float32) / 255.0
        return a[..., None] if a.ndim == 3 else a
    xtr, xva, xte = prep(d["train_images"]), prep(d["val_images"]), prep(d["test_images"])
    ytr, yva, yte = (d[k].ravel().astype(int) for k in ("train_labels", "val_labels", "test_labels"))
    C, ch = len(cfg["classes"]), cfg["channels"]

    # ---------------------------------------------------------------- train
    tf.keras.utils.set_random_seed(args.seed)
    model = build_model(xtr.shape[1:], C)
    counts = np.bincount(ytr, minlength=C)
    class_weight = {c: float(len(ytr) / (C * counts[c])) for c in range(C)}
    ds = (tf.data.Dataset.from_tensor_slices((xtr, ytr)).shuffle(min(len(xtr), 4096), seed=args.seed)
          .map(augment_fn(ch == 3), num_parallel_calls=tf.data.AUTOTUNE).batch(args.batch)
          .prefetch(tf.data.AUTOTUNE))
    model.compile(tf.keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", ["accuracy"])
    cbs = [tf.keras.callbacks.EarlyStopping("val_loss", patience=7, restore_best_weights=True),
           tf.keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=3, min_lr=1e-5)]
    t0 = time.time()
    h = model.fit(ds, validation_data=(xva, yva), epochs=args.epochs or cfg["epochs"],
                  class_weight=class_weight, callbacks=cbs, verbose=2)
    fit_time = time.time() - t0
    hist = {k: [float(v) for v in vals] for k, vals in h.history.items() if k != "lr"}
    model.save(os.path.join(out_dir, "model.h5"))

    ex = X.explainable_from_sequential(model, "last_conv", name=cfg["title"])

    # ---------------------------------------------------------------- predictions & calibration
    p_val, p_te = ex.predict(xva), ex.predict(xte)
    lg_val, lg_te = np.log(p_val + 1e-12), np.log(p_te + 1e-12)
    T = X.fit_temperature(lg_val, yva)
    t1 = time.perf_counter()
    for _ in range(20):
        ex.predict(xte[:1])
    latency = (time.perf_counter() - t1) / 20 * 1000
    print(f"test acc {np.mean(p_te.argmax(1) == yte):.4f}  AUC {macro_auc(yte, p_te):.4f}  T={T:.2f}", flush=True)

    # ---------------------------------------------------------------- uncertainty
    mc = X.mc_dropout(ex, xte, T=args.mc_samples)
    mc_mean, mc_tot, mc_ale, mc_epi = X.decompose(mc)
    tta = X.tta_samples(ex, xte, n=8, flips=(ch == 3), seed=args.seed)
    tta_mean, tta_tot, tta_ale, tta_epi = X.decompose(tta)

    # ---------------------------------------------------------------- representation
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    e_tr, e_te = embed(model, xtr), embed(model, xte)
    rng = np.random.default_rng(args.seed)
    sub = np.sort(rng.choice(len(xte), min(len(xte), 1500), replace=False))
    tsne = TSNE(2, perplexity=30, init="pca", random_state=args.seed).fit_transform(e_te[sub])
    pca = PCA(2, random_state=args.seed).fit(e_tr)
    pca_te = pca.transform(e_te)
    sil = float(silhouette_score(e_te[sub], yte[sub]))

    # ---------------------------------------------------------------- OOD
    mus, prec, ood_table, id_scores, ood_scores = compute_ood(model, ex, name, args, cifar_test, xtr, ytr, xte, ch, C)

    # ---------------------------------------------------------------- robustness
    rob_idx = np.concatenate([np.where(yte == c)[0][:args.robust_per_class] for c in range(C)])
    xr, yr = xte[rob_idx], yte[rob_idx]
    clean_p = ex.predict(xr)
    robustness = {"clean": {"acc": float((clean_p.argmax(1) == yr).mean()), "auc": macro_auc(yr, clean_p)}}
    for kind in X.CORRUPTIONS:
        robustness[kind] = []
        for s in SEVERITIES:
            p = ex.predict(X.corrupt(xr, kind, s, seed=args.seed).astype(np.float32))
            robustness[kind].append({"acc": float((p.argmax(1) == yr).mean()), "auc": macro_auc(yr, p)})
    adv = []
    for e in FGSM_EPS:
        xa = (np.concatenate([X.fgsm(ex, xr[i:i + 100], yr[i:i + 100], e / 255.0) for i in range(0, len(xr), 100)])
              if e > 0 else xr)
        p = ex.predict(xa)
        adv.append({"eps": e, "acc": float((p.argmax(1) == yr).mean()), "auc": macro_auc(yr, p)})
    print("robustness done", flush=True)

    # ---------------------------------------------------------------- explanation faithfulness
    fidx = np.concatenate([np.where(yte == c)[0][:max(3, args.faith_n // C)] for c in range(C)])
    methods = ["Grad-CAM", "Grad-CAM++", "Saliency", "SmoothGrad", "Integrated Gradients", "Occlusion", "Random"]
    faith = {m: {"del": [], "ins": [], "del_curve": [], "ins_curve": []} for m in methods}
    for i in fidx:
        x = xte[i]
        maps, cls = X.all_maps(ex, x, fast=True)
        maps["Random"] = rng.random(x.shape[:2]).astype(np.float32)
        for m in methods:
            fr, dc, ic, da, ia = X.deletion_insertion(ex, x, maps[m], cls)
            faith[m]["del"].append(da)
            faith[m]["ins"].append(ia)
            faith[m]["del_curve"].append(dc.tolist())
            faith[m]["ins_curve"].append(ic.tolist())
    faith_summary = {m: {"deletion_auc": float(np.mean(v["del"])), "deletion_sd": float(np.std(v["del"])),
                         "insertion_auc": float(np.mean(v["ins"])), "insertion_sd": float(np.std(v["ins"])),
                         "del_curve": np.mean(v["del_curve"], 0).tolist(),
                         "ins_curve": np.mean(v["ins_curve"], 0).tolist(),
                         "del_all": v["del"], "ins_all": v["ins"]} for m, v in faith.items()}
    print("faithfulness done", flush=True)

    # ---------------------------------------------------------------- image-quality strata
    q = []
    for x in xte:
        g = X.to_gray(x)
        qi = X.quality_indices(g)
        q.append([g.mean(), qi["RMS contrast"], qi["Sharpness (Laplacian var.)"], qi["Noise σ (Immerkær)"]])
    q = np.array(q, np.float32)

    # ---------------------------------------------------------------- gallery for live XAI
    gal_idx = np.concatenate([np.where(yte == c)[0][:args.gallery_per_class] for c in range(C)])
    gallery = (xte[gal_idx] * 255).round().astype(np.uint8)

    np.savez_compressed(
        os.path.join(out_dir, "analysis.npz"),
        y_test=yte.astype(np.int16), y_val=yva.astype(np.int16),
        p_test=p_te.astype(np.float32), p_val=p_val.astype(np.float32),
        mc_mean=mc_mean.astype(np.float32), mc_total=mc_tot.astype(np.float32),
        mc_aleatoric=mc_ale.astype(np.float32), mc_epistemic=mc_epi.astype(np.float32),
        tta_mean=tta_mean.astype(np.float32), tta_total=tta_tot.astype(np.float32),
        tta_epistemic=tta_epi.astype(np.float32),
        tsne=tsne.astype(np.float32), tsne_idx=sub.astype(np.int32), pca=pca_te.astype(np.float32),
        maha_mus=mus.astype(np.float32), maha_prec=prec.astype(np.float32),
        quality=q, gallery=gallery, gallery_idx=gal_idx.astype(np.int32), gallery_y=yte[gal_idx].astype(np.int16),
        **{f"ood_id__{m}": v.astype(np.float32) for m, v in id_scores.items()},
        **{f"ood__{k}__{m}": v.astype(np.float32) for k, s in ood_scores.items() for m, v in s.items()},
    )
    summary = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": cfg["title"], "modality": cfg["modality"], "source": cfg["source"],
            "classes": cfg["classes"], "positive_class": cfg["positive"], "image_size": 64,
            "channels": ch, "n_train": int(len(ytr)), "n_val": int(len(yva)), "n_test": int(len(yte)),
            "train_counts": counts.tolist(), "test_counts": np.bincount(yte, minlength=C).tolist(),
            "params": int(model.count_params()), "fit_time_s": round(fit_time, 1),
            "latency_ms": round(latency, 2), "epochs_run": len(hist["loss"]),
            "best_epoch": int(np.argmin(hist["val_loss"])) + 1, "seed": args.seed,
            "architecture": "4×Conv(32-64-128-128)+BN, 2×MaxPool, SpatialDropout(0.1), GAP, "
                            "Dense(128) embedding, Dropout(0.4), softmax",
            "training": "Adam 1e-3, batch 64, class-weighted CE, ReduceLROnPlateau, early stopping "
                        "(val loss, patience 7), augmentation: brightness/contrast/translation"
                        + (" + flips/rot90" if ch == 3 else ""),
            "mc_samples": args.mc_samples, "tensorflow": tf.__version__,
            "faithfulness_images": int(len(fidx)), "robustness_images": int(len(rob_idx)),
            "silhouette_embedding": sil, "tsne_n": int(len(sub)),
        },
        "history": hist, "temperature": T,
        "ood": ood_table, "robustness": robustness, "fgsm": adv, "faithfulness": faith_summary,
        "faithfulness_fractions": np.linspace(0, 1, 21).tolist(),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    tf.keras.backend.clear_session()
    print(f"saved → {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--cifar-npz", default=None, help="local CIFAR-10 npz (x_test) for the far-OOD set")
    ap.add_argument("--out", default="results/medical")
    ap.add_argument("--datasets", default="pneumoniamnist,bloodmnist")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--mc-samples", type=int, default=30)
    ap.add_argument("--robust-per-class", type=int, default=150)
    ap.add_argument("--faith-n", type=int, default=48)
    ap.add_argument("--gallery-per-class", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ood-only", action="store_true", help="recompute OOD scores for saved models only")
    args = ap.parse_args()
    if args.cifar_npz:
        cifar_test = np.load(args.cifar_npz)["x_test"]
    else:
        cifar_test = tf.keras.datasets.cifar10.load_data()[1][0]
    cifar_test = cifar_test[np.random.default_rng(args.seed).permutation(len(cifar_test))]
    for name in args.datasets.split(","):
        (ood_only if args.ood_only else run)(name.strip(), args, cifar_test)


if __name__ == "__main__":
    main()
