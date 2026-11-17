"""
Shared explainability, uncertainty and image-analysis utilities.

Used by app.py (CIFAR-10), pages/2_Medical_Imaging.py and benchmark/train_medical.py.

Every explainer works on an `Explainable` wrapper that splits a classifier into
    feature_fn(x01) -> A        activations of the target conv layer (B, h, w, k)
    head_fn(A)      -> probs    the rest of the network (B, C)
where x01 is the raw image batch in [0, 1] (B, H, W, ch). Pre-processing lives
inside feature_fn, so every attribution map is expressed in input-pixel space.

Methods
    Grad-CAM (Selvaraju et al., 2017)          Grad-CAM++ (Chattopadhay et al., 2018)
    Vanilla saliency (Simonyan et al., 2014)    SmoothGrad (Smilkov et al., 2017)
    Integrated Gradients (Sundararajan 2017)    Occlusion sensitivity (Zeiler & Fergus, 2014)
    Deletion / insertion faithfulness (Petsiuk et al., RISE 2018)
    MC-Dropout (Gal & Ghahramani, 2016) and test-time augmentation uncertainty,
    decomposed into total = aleatoric (expected entropy) + epistemic (mutual information)
    First-order + GLCM texture features (IBSI-style definitions) and image-quality indices
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import tensorflow as tf
from scipy import ndimage

EPS = 1e-12


# =========================================================================== wrapper
@dataclass
class Explainable:
    feature_fn: Callable          # x01 -> conv activations
    head_fn: Callable             # activations -> probabilities
    head_mc_fn: Optional[Callable] = None   # activations -> probs with dropout ON (None = no dropout)
    name: str = "model"

    def predict(self, x01, batch=128):
        out = []
        for i in range(0, len(x01), batch):
            xb = tf.convert_to_tensor(x01[i:i + batch], tf.float32)
            out.append(self.head_fn(self.feature_fn(xb)).numpy())
        return np.concatenate(out)

    def full(self, x):
        return self.head_fn(self.feature_fn(x))


def _run_layers(layers, x, mc=False):
    for layer in layers:
        if mc and isinstance(layer, (tf.keras.layers.Dropout, tf.keras.layers.SpatialDropout2D)):
            x = layer(x, training=True)
        else:
            x = layer(x, training=False)
    return x


def explainable_from_sequential(model, cam_layer: Optional[str] = None, preprocess=None, name="model"):
    """Split a linear (Sequential) Keras model at `cam_layer` (default: last Conv2D)."""
    layers = [l for l in model.layers if not isinstance(l, tf.keras.layers.InputLayer)]
    if cam_layer is None:
        cam_layer = [l.name for l in layers if isinstance(l, tf.keras.layers.Conv2D)][-1]
    k = [l.name for l in layers].index(cam_layer)
    pre, post = layers[:k + 1], layers[k + 1:]
    has_dropout = any(isinstance(l, (tf.keras.layers.Dropout, tf.keras.layers.SpatialDropout2D)) for l in layers)
    prep = preprocess or (lambda x: x)

    def feature_fn(x):
        return _run_layers(pre, prep(x))

    def head_fn(a):
        return _run_layers(post, a)

    def head_mc_fn(a):
        return _run_layers(post, a, mc=True)

    # dropout located *before* the CAM layer would not be sampled; acceptable for our models
    return Explainable(feature_fn, head_fn, head_mc_fn if has_dropout else None, name)


# =========================================================================== attribution maps
def _norm(m):
    m = np.asarray(m, np.float64)
    m = m - m.min()
    return (m / (m.max() + EPS)).astype(np.float32)


def _upsample(cam, size):
    """Bilinear upsampling of a (h, w) map to (H, W)."""
    t = tf.image.resize(cam[None, ..., None].astype(np.float32), size, method="bilinear")
    return t.numpy()[0, ..., 0]


def grad_cam(ex: Explainable, x01, class_idx=None, plus_plus=False):
    x = tf.convert_to_tensor(x01[None] if x01.ndim == 3 else x01, tf.float32)
    with tf.GradientTape(persistent=plus_plus) as tape:
        a = ex.feature_fn(x)
        tape.watch(a)
        p = ex.head_fn(a)
        if class_idx is None:
            class_idx = int(tf.argmax(p[0]))
        score = p[:, class_idx]
    g = tape.gradient(score, a)[0].numpy()          # (h, w, k)
    a = a[0].numpy()
    if not plus_plus:
        w = g.mean(axis=(0, 1))
    else:
        # Grad-CAM++ closed form with exp(score) approximation (Chattopadhay et al., eq. 19)
        g2, g3 = g ** 2, g ** 3
        denom = 2 * g2 + (a * g3).sum(axis=(0, 1), keepdims=True)
        alpha = g2 / np.where(np.abs(denom) > EPS, denom, EPS)
        w = (alpha * np.maximum(g, 0)).sum(axis=(0, 1))
    cam = np.maximum((a * w).sum(-1), 0)
    return _norm(_upsample(cam, x.shape[1:3])), class_idx


def _grad(ex, x, class_idx):
    x = tf.convert_to_tensor(x, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        p = ex.full(x)[:, class_idx]
    return tape.gradient(p, x).numpy()


def saliency(ex, x01, class_idx):
    g = _grad(ex, x01[None], class_idx)[0]
    return _norm(np.abs(g).max(-1))


def smoothgrad(ex, x01, class_idx, n=24, sigma=0.12, seed=0):
    rng = np.random.default_rng(seed)
    xs = np.clip(x01[None] + rng.normal(0, sigma, (n,) + x01.shape), 0, 1).astype(np.float32)
    g = _grad(ex, xs, class_idx)
    return _norm(np.abs(g).mean(0).max(-1))


def integrated_gradients(ex, x01, class_idx, steps=32, baseline=None):
    base = np.zeros_like(x01) if baseline is None else baseline
    alphas = np.linspace(0, 1, steps + 1)[1:, None, None, None]
    path = (base[None] + alphas * (x01 - base)[None]).astype(np.float32)
    g = _grad(ex, path, class_idx).mean(0)
    ig = (x01 - base) * g
    return _norm(np.abs(ig).sum(-1)), ig


def occlusion(ex, x01, class_idx, patch=None, stride=None, fill=None):
    H, W = x01.shape[:2]
    patch = patch or max(4, H // 8)
    stride = stride or max(2, patch // 2)
    fill = x01.mean(axis=(0, 1)) if fill is None else fill
    base = ex.predict(x01[None])[0, class_idx]
    heat, count, batch, coords = np.zeros((H, W)), np.zeros((H, W)), [], []
    for i in range(0, H - patch + 1, stride):
        for j in range(0, W - patch + 1, stride):
            xo = x01.copy()
            xo[i:i + patch, j:j + patch] = fill
            batch.append(xo)
            coords.append((i, j))
    probs = ex.predict(np.stack(batch).astype(np.float32))[:, class_idx]
    for (i, j), p in zip(coords, probs):
        heat[i:i + patch, j:j + patch] += base - p
        count[i:i + patch, j:j + patch] += 1
    heat = heat / np.maximum(count, 1)
    return _norm(np.maximum(heat, 0))


def all_maps(ex, x01, class_idx=None, fast=False):
    """Return {method: map(H, W)} for one image."""
    cam, class_idx = grad_cam(ex, x01, class_idx)
    out = {"Grad-CAM": cam,
           "Grad-CAM++": grad_cam(ex, x01, class_idx, plus_plus=True)[0],
           "Saliency": saliency(ex, x01, class_idx),
           "SmoothGrad": smoothgrad(ex, x01, class_idx, n=12 if fast else 24),
           "Integrated Gradients": integrated_gradients(ex, x01, class_idx, steps=16 if fast else 32)[0],
           "Occlusion": occlusion(ex, x01, class_idx, stride=max(4, x01.shape[0] // 8) if fast else None)}
    return out, class_idx


# =========================================================================== faithfulness
def deletion_insertion(ex, x01, heatmap, class_idx, steps=20, blur_sigma=None):
    """Petsiuk et al. (2018). Returns (fractions, deletion_curve, insertion_curve, del_auc, ins_auc).

    Deletion: pixels are replaced by a blurred copy in order of decreasing importance
    (a fast drop = faithful map → low AUC). Insertion: start from the blurred copy and
    restore pixels (a fast rise = faithful map → high AUC)."""
    H, W = x01.shape[:2]
    sigma = blur_sigma or H / 8
    blurred = np.stack([ndimage.gaussian_filter(x01[..., c], sigma) for c in range(x01.shape[-1])], -1)
    order = np.argsort(-heatmap.ravel())
    fr = np.linspace(0, 1, steps + 1)
    dels, inss = [], []
    for f in fr:
        k = int(round(f * H * W))
        mask = np.zeros(H * W, bool)
        mask[order[:k]] = True
        mask = mask.reshape(H, W, 1)
        dels.append(np.where(mask, blurred, x01))
        inss.append(np.where(mask, x01, blurred))
    p = ex.predict(np.stack(dels + inss).astype(np.float32))[:, class_idx]
    d, i = p[:steps + 1], p[steps + 1:]
    return fr, d, i, float(np.trapz(d, fr)), float(np.trapz(i, fr))


def pointing_game(heatmap, mask):
    """1 if the max of the heat-map falls inside a binary ROI mask."""
    r, c = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return int(mask[r, c] > 0)


# =========================================================================== uncertainty
def decompose(prob_samples):
    """prob_samples: (T, N, C). Returns mean probs, total, aleatoric, epistemic (nats)."""
    p_mean = prob_samples.mean(0)
    total = -(p_mean * np.log(p_mean + EPS)).sum(-1)
    aleatoric = -(prob_samples * np.log(prob_samples + EPS)).sum(-1).mean(0)
    epistemic = np.maximum(total - aleatoric, 0)
    return p_mean, total, aleatoric, epistemic


def mc_dropout(ex: Explainable, x01, T=30, batch=256):
    if ex.head_mc_fn is None:
        return None
    feats = []
    for i in range(0, len(x01), batch):
        feats.append(ex.feature_fn(tf.convert_to_tensor(x01[i:i + batch], tf.float32)))
    feats = tf.concat(feats, 0)
    samples = np.stack([ex.head_mc_fn(feats).numpy() for _ in range(T)])
    return samples


def tta_samples(ex: Explainable, x01, n=16, seed=0, flips=True, max_shift=0.08, noise=0.02):
    """Test-time augmentation samples (aleatoric/input uncertainty): flips, shifts, noise."""
    rng = np.random.default_rng(seed)
    x01 = x01 if x01.ndim == 4 else x01[None]
    H = x01.shape[1]
    out = []
    for t in range(n):
        x = x01.copy()
        if flips and t % 2 == 1:
            x = x[:, :, ::-1]
        s = rng.integers(-int(max_shift * H), int(max_shift * H) + 1, 2)
        x = np.roll(x, tuple(s), axis=(1, 2))
        if t > 0:
            x = np.clip(x + rng.normal(0, noise, x.shape), 0, 1)
        out.append(ex.predict(x.astype(np.float32)))
    return np.stack(out)


# =========================================================================== calibration
def fit_temperature(logits, y, grid=np.exp(np.linspace(np.log(0.05), np.log(10), 500))):
    """Temperature scaling (Guo et al., 2017) by NLL grid search on a validation set."""
    best_t, best = 1.0, np.inf
    for t in grid:
        z = logits / t
        z = z - z.max(1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(1, keepdims=True))
        nll = -logp[np.arange(len(y)), y].mean()
        if nll < best:
            best, best_t = nll, t
    return float(best_t)


def softmax_t(logits, t=1.0):
    z = logits / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def calibration_stats(p, y, n_bins=15):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, mce, rows = 0.0, 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            gap = abs((pred[m] == y[m]).mean() - conf[m].mean())
            ece += m.mean() * gap
            mce = max(mce, gap)
            rows.append((float(conf[m].mean()), float((pred[m] == y[m]).mean()), int(m.sum())))
    onehot = np.eye(p.shape[1])[y]
    brier = float(((p - onehot) ** 2).sum(1).mean())
    nll = float(-np.log(p[np.arange(len(y)), y] + EPS).mean())
    return dict(ece=float(ece), mce=float(mce), brier=brier, nll=nll, bins=rows)


# =========================================================================== selective prediction
def risk_coverage(confidence, correct):
    """Sort by confidence (high first) and return coverage, selective risk and AURC."""
    order = np.argsort(-confidence)
    c = correct[order].astype(float)
    n = len(c)
    cov = np.arange(1, n + 1) / n
    risk = 1 - np.cumsum(c) / np.arange(1, n + 1)
    aurc = float(np.trapz(risk, cov))
    acc = c.mean()
    # optimal AURC for this error rate (Geifman et al., 2019) → E-AURC
    r_star = 1 - acc
    opt = r_star + (1 - r_star) * np.log(1 - r_star + EPS) if r_star > 0 else 0.0
    return cov, risk, aurc, float(aurc - opt)


# =========================================================================== OOD
def energy_score(logits, t=1.0):
    """Negative free energy (Liu et al., 2020): higher = more in-distribution."""
    z = logits / t
    m = z.max(1)
    return t * (m + np.log(np.exp(z - m[:, None]).sum(1)))


def fit_mahalanobis(feats, y, n_classes):
    mus = np.stack([feats[y == c].mean(0) for c in range(n_classes)])
    centred = feats - mus[y]
    cov = centred.T @ centred / len(feats) + 1e-3 * np.eye(feats.shape[1])
    return mus, np.linalg.inv(cov)


def mahalanobis_score(feats, mus, prec):
    """Lee et al. (2018): negative min class-conditional Mahalanobis distance (higher = ID)."""
    d = np.stack([np.einsum("ij,jk,ik->i", feats - m, prec, feats - m) for m in mus], 1)
    return -d.min(1)


def ood_metrics(score_id, score_ood):
    """AUROC (ID positive) and FPR at 95% TPR. Higher score = more in-distribution."""
    from sklearn.metrics import roc_auc_score, roc_curve
    y = np.r_[np.ones(len(score_id)), np.zeros(len(score_ood))]
    s = np.r_[score_id, score_ood]
    auroc = roc_auc_score(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    fpr95 = float(fpr[np.searchsorted(tpr, 0.95)])
    return float(auroc), fpr95


# =========================================================================== corruptions
def corrupt(x01, kind, severity, seed=0):
    """Hendrycks & Dietterich (2019)-style corruptions on a (N,H,W,C) batch in [0,1]. severity 1..5."""
    rng = np.random.default_rng(seed)
    s = severity
    if kind == "Gaussian noise":
        return np.clip(x01 + rng.normal(0, [0.02, 0.04, 0.06, 0.09, 0.12][s - 1], x01.shape), 0, 1)
    if kind == "Gaussian blur":
        sig = [0.5, 0.8, 1.1, 1.5, 2.0][s - 1]
        return np.stack([ndimage.gaussian_filter(x, (sig, sig, 0)) for x in x01])
    if kind == "Contrast":
        c = [0.75, 0.6, 0.45, 0.3, 0.2][s - 1]
        m = x01.mean(axis=(1, 2, 3), keepdims=True)
        return np.clip((x01 - m) * c + m, 0, 1)
    if kind == "Brightness":
        return np.clip(x01 + [0.08, 0.15, 0.22, 0.3, 0.4][s - 1], 0, 1)
    if kind == "JPEG":
        from io import BytesIO
        from PIL import Image
        q = [60, 40, 25, 15, 8][s - 1]
        out = []
        for x in x01:
            arr = (x * 255).round().astype(np.uint8)
            img = Image.fromarray(arr[..., 0] if arr.shape[-1] == 1 else arr)
            buf = BytesIO()
            img.save(buf, "JPEG", quality=q)
            r = np.asarray(Image.open(BytesIO(buf.getvalue()))).astype(np.float32) / 255
            out.append(r[..., None] if r.ndim == 2 else r)
        return np.stack(out)
    if kind == "Salt & pepper":
        p = [0.01, 0.02, 0.04, 0.06, 0.1][s - 1]
        x = x01.copy()
        m = rng.random(x.shape[:3])
        x[m < p / 2] = 0
        x[m > 1 - p / 2] = 1
        return x
    raise ValueError(kind)


CORRUPTIONS = ["Gaussian noise", "Gaussian blur", "Contrast", "Brightness", "JPEG", "Salt & pepper"]


def fgsm(ex: Explainable, x01, y, eps):
    """Fast Gradient Sign Method (Goodfellow et al., 2015) on the true label."""
    x = tf.convert_to_tensor(x01, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        p = ex.full(x)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, p)
    g = tape.gradient(loss, x)
    return tf.clip_by_value(x + eps * tf.sign(g), 0, 1).numpy()


# =========================================================================== radiomics / image analysis
def to_gray(img):
    img = np.asarray(img, np.float64)
    if img.ndim >= 3 and img.shape[-1] == 3:
        return img @ np.array([0.299, 0.587, 0.114])
    if img.ndim >= 3 and img.shape[-1] == 1:
        return img[..., 0]
    return img


def first_order(gray01, bins=64):
    g = gray01.ravel()
    h, _ = np.histogram(g, bins=bins, range=(0, 1))
    p = h / h.sum()
    mu, sd = g.mean(), g.std()
    return {
        "Mean": mu, "Std": sd, "Median": float(np.median(g)),
        "P10": float(np.percentile(g, 10)), "P90": float(np.percentile(g, 90)),
        "IQR": float(np.percentile(g, 75) - np.percentile(g, 25)),
        "Skewness": float(((g - mu) ** 3).mean() / (sd ** 3 + EPS)),
        "Kurtosis": float(((g - mu) ** 4).mean() / (sd ** 4 + EPS)),
        "Entropy (bits)": float(-(p[p > 0] * np.log2(p[p > 0])).sum()),
        "Uniformity": float((p ** 2).sum()),
    }


def glcm_features(gray01, levels=32, distances=(1, 2), angles=(0, 45, 90, 135)):
    """Symmetric, normalised grey-level co-occurrence matrix averaged over offsets."""
    q = np.clip((gray01 * levels).astype(int), 0, levels - 1)
    H, W = q.shape
    P = np.zeros((levels, levels))
    for d in distances:
        for a in angles:
            dy, dx = {0: (0, d), 45: (-d, d), 90: (-d, 0), 135: (-d, -d)}[a]
            r0, r1 = max(0, -dy), H - max(0, dy)
            c0, c1 = max(0, -dx), W - max(0, dx)
            i = q[r0:r1, c0:c1].ravel()
            j = q[r0 + dy:r1 + dy, c0 + dx:c1 + dx].ravel()
            np.add.at(P, (i, j), 1)
    P = P + P.T
    P /= P.sum()
    I, J = np.indices(P.shape)
    mi, mj = (I * P).sum(), (J * P).sum()
    si = np.sqrt(((I - mi) ** 2 * P).sum())
    sj = np.sqrt(((J - mj) ** 2 * P).sum())
    return {
        "GLCM contrast": float(((I - J) ** 2 * P).sum()),
        "GLCM dissimilarity": float((np.abs(I - J) * P).sum()),
        "GLCM homogeneity": float((P / (1 + (I - J) ** 2)).sum()),
        "GLCM energy": float(np.sqrt((P ** 2).sum())),
        "GLCM correlation": float(((I - mi) * (J - mj) * P).sum() / (si * sj + EPS)),
        "GLCM entropy": float(-(P[P > 0] * np.log2(P[P > 0])).sum()),
    }, P


def quality_indices(gray01):
    lap = ndimage.laplace(gray01)
    # Immerkær (1996) fast noise-sigma estimate
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
    H, W = gray01.shape
    sigma_n = np.sqrt(np.pi / 2) / (6 * (W - 2) * (H - 2)) * np.abs(ndimage.convolve(gray01, k))[1:-1, 1:-1].sum()
    gx, gy = ndimage.sobel(gray01, 0), ndimage.sobel(gray01, 1)
    return {
        "Sharpness (Laplacian var.)": float(lap.var()),
        "RMS contrast": float(gray01.std()),
        "Noise σ (Immerkær)": float(sigma_n),
        "SNR (mean/σ_noise)": float(gray01.mean() / (sigma_n + EPS)),
        "Edge density (Sobel)": float((np.hypot(gx, gy) > 0.25).mean()),
        "Dynamic range": float(gray01.max() - gray01.min()),
        "Saturated px (%)": float(((gray01 < 0.02) | (gray01 > 0.98)).mean() * 100),
    }


def equalize(gray01, bins=256):
    h, edges = np.histogram(gray01.ravel(), bins, (0, 1))
    cdf = h.cumsum() / h.sum()
    return np.interp(gray01.ravel(), edges[:-1], cdf).reshape(gray01.shape)


def overlay(img01, heat, alpha=0.45):
    """RGB overlay of a heat-map (jet-like) on an image."""
    rgb = img01 if img01.shape[-1] == 3 else np.repeat(img01, 3, -1)
    return np.clip((1 - alpha) * rgb + alpha * jet(heat), 0, 1)


def jet(v):
    """Numpy 'jet' colour map (no matplotlib dependency)."""
    v = np.clip(v, 0, 1)[..., None]
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.concatenate([r, g, b], -1)
