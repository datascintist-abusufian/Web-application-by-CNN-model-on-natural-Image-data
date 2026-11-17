"""
Medical Imaging Analysis — a research-grade evaluation page for two MedMNIST v2 tasks.

    PneumoniaMNIST  paediatric chest X-ray, normal vs pneumonia (binary)
    BloodMNIST      blood-cell microscopy, 8 cell types (multi-class)

Pre-computed artefacts (benchmark/train_medical.py) live in results/medical/<dataset>/.
Population-level analyses read those files; the case-level tabs (explainability, image
analysis, out-of-distribution check of an upload) run the model live.

For research and teaching only. Not a medical device and not for clinical decisions.
"""
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from plotly.subplots import make_subplots
from scipy.stats import norm, spearmanr
from sklearn.metrics import (average_precision_score, cohen_kappa_score, confusion_matrix,
                             matthews_corrcoef, precision_recall_curve, roc_auc_score, roc_curve)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Medical Imaging · XAI & Clinical Evaluation", page_icon="🩻", layout="wide")

MED = ROOT / "results" / "medical"
DATASETS = {"PneumoniaMNIST — chest X-ray": "pneumoniamnist", "BloodMNIST — blood-cell microscopy": "bloodmnist"}
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a5cd6", "#d63a7a", "#3bb3c4", "#7a7a72"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
METHOD_COLOR = {"Grad-CAM": "#eb6834", "Grad-CAM++": "#d63a7a", "Saliency": "#2a78d6", "SmoothGrad": "#3bb3c4",
                "Integrated Gradients": "#1baf7a", "Occlusion": "#eda100", "Random": "#7a7a72"}

st.markdown("""
<style>
.gradient-header{background:linear-gradient(135deg,#0f766e 0%,#1e3a8a 100%);color:#fff;
  padding:1.6rem 2rem;border-radius:15px;margin:.5rem 0 1rem;text-align:center}
.gradient-header h1{color:#fff;margin:0;font-size:2rem}
.gradient-header p{margin:.35rem 0 0;opacity:.92}
.kpi{background:#f8f9fa;border-radius:12px;padding:1rem 1.1rem;border-left:5px solid #0f766e;height:100%}
.kpi .lbl{font-size:.78rem;color:#52514e;text-transform:uppercase;letter-spacing:.04em}
.kpi .val{font-size:1.5rem;font-weight:700;color:#0b0b0b;line-height:1.2;margin-top:.2rem}
.kpi .sub{font-size:.8rem;color:#52514e}
.finding{background:#f8f9fa;border-radius:10px;padding:.75rem 1rem;margin:.35rem 0;border-left:4px solid #17a2b8}
.finding.warn{border-left-color:#dc3545;background:#fff5f5}
.finding.ok{border-left-color:#1baf7a;background:#f3fbf7}
.small-note{font-size:.82rem;color:#52514e}
.disclaimer{font-size:.82rem;color:#7a1f1f;background:#fff5f5;border-radius:8px;padding:.5rem .8rem;border:1px solid #f3c7c7}
</style>
""", unsafe_allow_html=True)


# =========================================================================== helpers
def style_fig(fig, height=420, legend=True):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=55, b=10),
                      font=dict(color=INK, size=13), showlegend=legend,
                      legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0, title=None,
                                  font=dict(color=INK2, size=12)))
    if fig.layout.title.text:
        fig.update_layout(title=dict(y=0.98, yanchor="top", x=0, xanchor="left", font=dict(size=16)))
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def show(fig):
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="lbl">{label}</div><div class="val">{value}</div>'
                 f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def finding(text, kind=""):
    st.markdown(f'<div class="finding {kind}">{text}</div>', unsafe_allow_html=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def fmt_ci(v, lo, hi, pct=True):
    if pct:
        return f"{v * 100:.1f}% ({lo * 100:.1f}–{hi * 100:.1f})"
    return f"{v:.3f} ({lo:.3f}–{hi:.3f})"


def big(img_u8, size=192):
    a = img_u8[..., 0] if img_u8.ndim == 3 and img_u8.shape[-1] == 1 else img_u8
    return Image.fromarray(a).resize((size, size), Image.NEAREST)


def to01_u8(x01):
    return (np.clip(x01, 0, 1) * 255).round().astype(np.uint8)


# =========================================================================== data
@st.cache_data(show_spinner=False)
def load_dataset(key, mtime):
    d = MED / key
    with open(d / "summary.json") as f:
        s = json.load(f)
    npz = np.load(d / "analysis.npz")
    arr = {k: npz[k] for k in npz.files}
    return s, arr


@st.cache_resource(show_spinner="Loading model and TensorFlow…")
def load_model(key):
    import tensorflow as tf
    import xai_utils as X
    model = tf.keras.models.load_model(MED / key / "model.h5", compile=False)
    return model, X.explainable_from_sequential(model, "last_conv", name=key)


@st.cache_data(show_spinner=False)
def bootstrap_auc(y, s, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            out.append(roc_auc_score(y[i], s[i]))
    return np.percentile(out, [2.5, 97.5])


@st.cache_data(show_spinner=False)
def bootstrap_metric(y, p, which, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        pred = p[i].argmax(1)
        if which == "mcc":
            vals.append(matthews_corrcoef(y[i], pred))
        elif which == "kappa":
            vals.append(cohen_kappa_score(y[i], pred))
        elif which == "macro_auc":
            try:
                vals.append(roc_auc_score(y[i], p[i], multi_class="ovr", average="macro"))
            except ValueError:
                pass
    return np.percentile(vals, [2.5, 97.5])


def binary_metrics(y, s, thr):
    pred = (s >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    ppv, npv = tp / max(tp + fp, 1), tn / max(tn + fn, 1)
    lr_p = sens / max(1 - spec, 1e-9)
    lr_n = (1 - sens) / max(spec, 1e-9)
    return dict(tp=tp, fn=fn, tn=tn, fp=fp, sens=sens, spec=spec, ppv=ppv, npv=npv,
                acc=(tp + tn) / len(y), bacc=(sens + spec) / 2,
                f1=2 * tp / max(2 * tp + fp + fn, 1), mcc=matthews_corrcoef(y, pred),
                kappa=cohen_kappa_score(y, pred), lr_pos=lr_p, lr_neg=lr_n,
                dor=lr_p / max(lr_n, 1e-9), youden=sens + spec - 1)


def net_benefit(y, s, thresholds):
    n = len(y)
    nb, nb_all = [], []
    prev = y.mean()
    for t in thresholds:
        pred = s >= t
        tp, fp = (pred & (y == 1)).sum(), (pred & (y == 0)).sum()
        w = t / (1 - t)
        nb.append(tp / n - fp / n * w)
        nb_all.append(prev - (1 - prev) * w)
    return np.array(nb), np.array(nb_all)


def softmax_t(p, t):
    z = np.log(p + 1e-12) / t
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def calib(p, y, n_bins=15):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, mce, rows = 0.0, 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            gap = abs((pred[m] == y[m]).mean() - conf[m].mean())
            ece += m.mean() * gap
            mce = max(mce, gap)
            rows.append((conf[m].mean(), (pred[m] == y[m]).mean(), int(m.sum())))
    brier = ((p - np.eye(p.shape[1])[y]) ** 2).sum(1).mean()
    nll = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
    return ece, mce, brier, nll, rows


def risk_cov(conf, correct):
    o = np.argsort(-conf)
    c = correct[o].astype(float)
    n = len(c)
    cov = np.arange(1, n + 1) / n
    risk = 1 - np.cumsum(c) / np.arange(1, n + 1)
    return cov, risk, np.trapz(risk, cov)


# =========================================================================== header & dataset choice
st.markdown("""
<div class="gradient-header">
  <h1>🩻 Medical Imaging Analysis</h1>
  <p>Clinical evaluation · calibration · uncertainty · out-of-distribution detection · robustness · explainability · radiomics</p>
</div>""", unsafe_allow_html=True)

available = {k: v for k, v in DATASETS.items() if (MED / v / "summary.json").exists()}
if not available:
    st.error("No pre-computed results found in `results/medical/`. Run `python benchmark/train_medical.py` first.")
    st.stop()

with st.sidebar:
    st.header("🩺 Study set-up")
    ds_label = st.radio("Dataset", list(available), index=0)
    key = available[ds_label]
summary, A = load_dataset(key, (MED / key / "analysis.npz").stat().st_mtime)
meta = summary["meta"]
CLASSES = meta["classes"]
C = len(CLASSES)
BINARY = C == 2
POS = meta["positive_class"] if BINARY else None
y = A["y_test"].astype(int)
p_raw = A["p_test"].astype(np.float64)
T = summary["temperature"]

with st.sidebar:
    use_ts = st.toggle("Apply temperature scaling", value=True,
                       help=f"Post-hoc calibration fitted on the validation split (T = {T:.2f}). "
                            "Changes probabilities, never the predicted class.")
    p = softmax_t(p_raw, T) if use_ts else p_raw
    st.markdown("---")
    st.subheader("🔎 Case under analysis")
    st.caption("Used by the Explainability and Image-analysis tabs.")
    src = st.radio("Image source", ["Test-set gallery", "Upload an image"], horizontal=False)
    gallery, gal_y, gal_idx = A["gallery"], A["gallery_y"].astype(int), A["gallery_idx"]
    case_label, case_img = None, None
    if src == "Test-set gallery":
        options = list(range(len(gallery)))
        gi = st.selectbox("Gallery image", options,
                          format_func=lambda i: f"#{gal_idx[i]} · true: {CLASSES[gal_y[i]]}")
        case_img = gallery[gi].astype(np.float32) / 255
        case_label = int(gal_y[gi])
        st.image(big(gallery[gi], 160), caption=f"Ground truth: {CLASSES[case_label]}")
    else:
        up = st.file_uploader("Image (PNG/JPG/TIFF/BMP)", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
        if up is not None:
            im = Image.open(up)
            im = im.convert("L" if meta["channels"] == 1 else "RGB")
            w, h = im.size
            s = min(w, h)
            im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((64, 64), Image.BILINEAR)
            a = np.asarray(im).astype(np.float32) / 255
            case_img = a[..., None] if a.ndim == 2 else a
            st.image(big(to01_u8(case_img), 160), caption="Uploaded · centre-cropped, 64×64 (unlabelled)")
    st.markdown("---")
    st.markdown('<div class="disclaimer">Research and teaching tool. Not a medical device; '
                'not validated for diagnosis or patient management.</div>', unsafe_allow_html=True)

pred = p.argmax(1)
correct = pred == y
acc = correct.mean()
macro_auc = roc_auc_score(y, p[:, 1]) if BINARY else roc_auc_score(y, p, multi_class="ovr", average="macro")
ece_now = calib(p, y)[0]

# =========================================================================== KPI row
k = st.columns(6)
kpi(k[0], "Task", "Binary" if BINARY else f"{C}-class", meta["modality"].split("(")[0])
kpi(k[1], "Test images", f"{meta['n_test']:,}", f"train {meta['n_train']:,} · val {meta['n_val']:,}")
lo, hi = wilson(correct.sum(), len(y))
kpi(k[2], "Accuracy", f"{acc * 100:.1f}%", f"95% CI {lo * 100:.1f}–{hi * 100:.1f}")
kpi(k[3], "AUROC" if BINARY else "Macro AUROC", f"{macro_auc:.3f}", "one-vs-rest" if not BINARY else "pneumonia vs normal")
kpi(k[4], "ECE", f"{ece_now:.3f}", "temperature-scaled" if use_ts else "uncalibrated")
kpi(k[5], "Model", f"{meta['params'] / 1e3:.0f} k params", f"{meta['latency_ms']:.1f} ms / image (CPU)")
st.markdown("")

tabs = st.tabs(["🩺 Overview", "📋 Clinical performance", "📈 Decision curves", "🎯 Calibration",
                "🎲 Uncertainty & referral", "🚨 Out-of-distribution", "🛡️ Robustness & stratification",
                "🧬 Representations", "🔬 Explainability (XAI)", "🧪 Image analysis / radiomics", "📑 Model card"])

# =========================================================================== 1. overview
with tabs[0]:
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown(f"**Dataset.** {meta['dataset']} — {meta['modality']}. Source: {meta['source']}. "
                    f"Images 64×64, {'grey-scale' if meta['channels'] == 1 else 'RGB'}; official MedMNIST v2 "
                    f"train/validation/test split (no patient overlap across splits as released).")
        st.markdown(f"**Model.** {meta['architecture']}. {meta['training']}. "
                    f"Best epoch {meta['best_epoch']} of {meta['epochs_run']}, trained in {meta['fit_time_s']:.0f} s on CPU.")
        dfc = pd.DataFrame({"class": CLASSES, "train": meta["train_counts"], "test": meta["test_counts"]})
        fig = go.Figure()
        for i, col in enumerate(["train", "test"]):
            fig.add_bar(x=dfc["class"], y=dfc[col] / dfc[col].sum() * 100, name=col,
                        marker_color=PALETTE[i], text=dfc[col], textposition="outside")
        fig.update_layout(title="Class distribution (% of split; counts on bars)", yaxis_title="%", barmode="group")
        show(style_fig(fig, 380))
        if BINARY:
            prev_tr = meta["train_counts"][POS] / sum(meta["train_counts"])
            prev_te = meta["test_counts"][POS] / sum(meta["test_counts"])
            finding(f"Prevalence of <b>{CLASSES[POS]}</b>: {prev_tr:.1%} in training vs {prev_te:.1%} in test. "
                    "Predictive values (PPV/NPV) depend on prevalence, so they will differ in a screening population "
                    "— see the prevalence slider in <i>Clinical performance</i>.", "warn" if abs(prev_tr - prev_te) > .05 else "")
    with c2:
        h = summary["history"]
        ep = np.arange(1, len(h["loss"]) + 1)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["Loss", "Accuracy"], vertical_spacing=.1)
        for i, (k_, nm) in enumerate([("loss", "train"), ("val_loss", "validation")]):
            fig.add_scatter(x=ep, y=h[k_], name=nm, line=dict(color=PALETTE[i]), row=1, col=1)
        for i, (k_, nm) in enumerate([("accuracy", "train"), ("val_accuracy", "validation")]):
            fig.add_scatter(x=ep, y=h[k_], name=nm, line=dict(color=PALETTE[i]), showlegend=False, row=2, col=1)
        fig.add_vline(x=meta["best_epoch"], line_dash="dot", line_color=INK2)
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        show(style_fig(fig, 470))
    st.markdown("##### Test-set gallery")
    per = len(gallery) // C
    for c in range(C):
        cols = st.columns(per + 1)
        cols[0].markdown(f"**{CLASSES[c]}**")
        for j, i in enumerate(np.where(gal_y == c)[0][:per]):
            pi = p[gal_idx[i]]
            ok = "✅" if pi.argmax() == c else "❌"
            cols[j + 1].image(big(gallery[i], 112), caption=f"{ok} {CLASSES[pi.argmax()]} {pi.max():.0%}")

# =========================================================================== 2. clinical performance
with tabs[1]:
    if BINARY:
        s = p[:, POS]
        fpr, tpr, thr = roc_curve(y, s)
        j = np.argmax(tpr - fpr)
        youden_t = float(np.clip(thr[j], 0, 1))
        i95 = np.searchsorted(tpr, 0.95)
        sens95_t = float(np.clip(thr[min(i95, len(thr) - 1)], 0, 1))
        c1, c2 = st.columns([1, 2])
        with c1:
            op_opts = ["Default (0.50)", f"Youden-optimal ({youden_t:.2f})",
                       f"High-sensitivity ≥95% ({sens95_t:.2f})", "Custom"]
            op = st.radio("Operating point", op_opts, index=1)
            t = {0: 0.5, 1: youden_t, 2: sens95_t}.get(op_opts.index(op), None)
            if t is None:
                t = st.slider("Decision threshold on P(pneumonia)", 0.01, 0.99, 0.5, 0.01)
            m = binary_metrics(y, s, t)
            cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
            fig = px.imshow(cm, x=[f"pred {c}" for c in CLASSES], y=[f"true {c}" for c in CLASSES],
                            text_auto=True, color_continuous_scale="Blues", title=f"Confusion matrix @ t = {t:.2f}")
            show(style_fig(fig, 340, legend=False))
        with c2:
            auc_lo, auc_hi = bootstrap_auc(y, s)
            npos, nneg = m["tp"] + m["fn"], m["tn"] + m["fp"]
            rows = [
                ("Sensitivity (recall, TPR)", fmt_ci(m["sens"], *wilson(m["tp"], npos))),
                ("Specificity (TNR)", fmt_ci(m["spec"], *wilson(m["tn"], nneg))),
                ("PPV (precision)", fmt_ci(m["ppv"], *wilson(m["tp"], m["tp"] + m["fp"]))),
                ("NPV", fmt_ci(m["npv"], *wilson(m["tn"], m["tn"] + m["fn"]))),
                ("Accuracy", fmt_ci(m["acc"], *wilson(m["tp"] + m["tn"], len(y)))),
                ("Balanced accuracy", f"{m['bacc'] * 100:.1f}%"),
                ("F1", f"{m['f1']:.3f}"),
                ("Matthews correlation (MCC)", f"{m['mcc']:.3f}"),
                ("Cohen's κ", f"{m['kappa']:.3f}"),
                ("Youden's J", f"{m['youden']:.3f}"),
                ("LR+ / LR−", f"{m['lr_pos']:.2f} / {m['lr_neg']:.3f}"),
                ("Diagnostic odds ratio", f"{m['dor']:.1f}"),
                ("AUROC (bootstrap 95% CI)", fmt_ci(roc_auc_score(y, s), auc_lo, auc_hi, pct=False)),
                ("Average precision (PR-AUC)", f"{average_precision_score(y, s):.3f}"),
            ]
            st.dataframe(pd.DataFrame(rows, columns=["Metric", "Value (95% CI)"]), hide_index=True,
                         use_container_width=True, height=530)
            st.caption("Proportions: Wilson score intervals. AUROC: 1,000 bootstrap resamples (percentile).")
        c1, c2, c3 = st.columns(3)
        with c1:
            fig = go.Figure()
            fig.add_scatter(x=fpr, y=tpr, name=f"ROC (AUC {roc_auc_score(y, s):.3f})", line=dict(color=PALETTE[0], width=2.5))
            fig.add_scatter(x=[0, 1], y=[0, 1], name="chance", line=dict(color=INK2, dash="dot"))
            fig.add_scatter(x=[1 - m["spec"]], y=[m["sens"]], mode="markers", name="operating point",
                            marker=dict(size=13, color=PALETTE[1], symbol="x"))
            fig.update_layout(title="ROC curve", xaxis_title="1 − specificity", yaxis_title="Sensitivity")
            show(style_fig(fig, 400))
        with c2:
            pr, rc, _ = precision_recall_curve(y, s)
            fig = go.Figure()
            fig.add_scatter(x=rc, y=pr, name=f"PR (AP {average_precision_score(y, s):.3f})", line=dict(color=PALETTE[2], width=2.5))
            fig.add_hline(y=y.mean(), line_dash="dot", line_color=INK2, annotation_text="prevalence")
            fig.add_scatter(x=[m["sens"]], y=[m["ppv"]], mode="markers", name="operating point",
                            marker=dict(size=13, color=PALETTE[1], symbol="x"))
            fig.update_layout(title="Precision–recall curve", xaxis_title="Recall (sensitivity)", yaxis_title="Precision (PPV)")
            show(style_fig(fig, 400))
        with c3:
            ts = np.linspace(0.01, 0.99, 99)
            sw = [binary_metrics(y, s, t_) for t_ in ts]
            fig = go.Figure()
            for i, (k_, nm) in enumerate([("sens", "Sensitivity"), ("spec", "Specificity"), ("ppv", "PPV"), ("npv", "NPV")]):
                fig.add_scatter(x=ts, y=[r[k_] for r in sw], name=nm, line=dict(color=PALETTE[i]))
            fig.add_vline(x=t, line_dash="dot", line_color=INK2)
            fig.update_layout(title="Threshold sweep", xaxis_title="Threshold", yaxis_title="Value")
            show(style_fig(fig, 400))
        st.markdown("##### Transportability: predictive values at a different prevalence")
        prev = st.slider("Target-population prevalence of pneumonia", 0.01, 0.9, 0.10, 0.01,
                         help="Bayes' theorem with the test-set sensitivity/specificity at the chosen threshold.")
        ppv_p = m["sens"] * prev / (m["sens"] * prev + (1 - m["spec"]) * (1 - prev))
        npv_p = m["spec"] * (1 - prev) / (m["spec"] * (1 - prev) + (1 - m["sens"]) * prev)
        pr_grid = np.linspace(0.01, 0.9, 90)
        fig = go.Figure()
        fig.add_scatter(x=pr_grid, y=m["sens"] * pr_grid / (m["sens"] * pr_grid + (1 - m["spec"]) * (1 - pr_grid)),
                        name="PPV", line=dict(color=PALETTE[2]))
        fig.add_scatter(x=pr_grid, y=m["spec"] * (1 - pr_grid) / (m["spec"] * (1 - pr_grid) + (1 - m["sens"]) * pr_grid),
                        name="NPV", line=dict(color=PALETTE[3]))
        fig.add_vline(x=prev, line_dash="dot", line_color=INK2)
        fig.add_vline(x=y.mean(), line_dash="dash", line_color=PALETTE[1], annotation_text="test prevalence")
        fig.update_layout(title=f"At prevalence {prev:.0%}: PPV {ppv_p:.1%}, NPV {npv_p:.1%}",
                          xaxis_title="Prevalence", yaxis_title="Predictive value")
        show(style_fig(fig, 360))
    else:
        cm = confusion_matrix(y, pred, labels=range(C))
        rows = []
        for c in range(C):
            tp = cm[c, c]; fn = cm[c].sum() - tp; fp = cm[:, c].sum() - tp; tn = cm.sum() - tp - fn - fp
            yb = (y == c).astype(int)
            auc_c = roc_auc_score(yb, p[:, c])
            lo_a, hi_a = bootstrap_auc(yb, p[:, c], n_boot=300)
            rows.append({"Class": CLASSES[c], "n": int(cm[c].sum()),
                         "Sensitivity": fmt_ci(tp / (tp + fn), *wilson(tp, tp + fn)),
                         "Specificity": fmt_ci(tn / (tn + fp), *wilson(tn, tn + fp)),
                         "PPV": fmt_ci(tp / max(tp + fp, 1), *wilson(tp, tp + fp)),
                         "NPV": fmt_ci(tn / max(tn + fn, 1), *wilson(tn, tn + fn)),
                         "F1": round(2 * tp / max(2 * tp + fp + fn, 1), 3),
                         "AUROC (95% CI)": fmt_ci(auc_c, lo_a, hi_a, pct=False)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        mcc_ci = bootstrap_metric(y, p, "mcc")
        kap_ci = bootstrap_metric(y, p, "kappa")
        auc_ci = bootstrap_metric(y, p, "macro_auc", n_boot=200)
        k2 = st.columns(4)
        kpi(k2[0], "Balanced accuracy", f"{np.mean(np.diag(cm) / cm.sum(1)):.1%}", "mean per-class sensitivity")
        kpi(k2[1], "MCC", f"{matthews_corrcoef(y, pred):.3f}", f"95% CI {mcc_ci[0]:.3f}–{mcc_ci[1]:.3f}")
        kpi(k2[2], "Cohen's κ", f"{cohen_kappa_score(y, pred):.3f}", f"95% CI {kap_ci[0]:.3f}–{kap_ci[1]:.3f}")
        kpi(k2[3], "Macro AUROC", f"{macro_auc:.3f}", f"95% CI {auc_ci[0]:.3f}–{auc_ci[1]:.3f}")
        c1, c2 = st.columns(2)
        with c1:
            cmn = cm / cm.sum(1, keepdims=True)
            fig = px.imshow(cmn, x=CLASSES, y=CLASSES, text_auto=".2f", color_continuous_scale="Blues", zmin=0, zmax=1,
                            title="Row-normalised confusion matrix (rows = truth)")
            fig.update_xaxes(tickangle=-35)
            show(style_fig(fig, 520, legend=False))
        with c2:
            fig = go.Figure()
            for c in range(C):
                f_, t_, _ = roc_curve((y == c).astype(int), p[:, c])
                fig.add_scatter(x=f_, y=t_, name=f"{CLASSES[c]} ({roc_auc_score((y == c), p[:, c]):.3f})",
                                line=dict(color=PALETTE[c % len(PALETTE)]))
            fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(color=INK2, dash="dot"), showlegend=False)
            fig.update_layout(title="One-vs-rest ROC", xaxis_title="1 − specificity", yaxis_title="Sensitivity")
            show(style_fig(fig, 520))
        off = cm.copy().astype(float)
        np.fill_diagonal(off, 0)
        top = np.dstack(np.unravel_index(np.argsort(-off.ravel())[:5], off.shape))[0]
        st.markdown("**Most frequent confusions:** " + " · ".join(
            f"{CLASSES[a]} → {CLASSES[b]} ({int(off[a, b])})" for a, b in top if off[a, b] > 0))

# =========================================================================== 3. decision curves
with tabs[2]:
    st.markdown("Decision curve analysis (Vickers & Elkin, 2006) asks whether acting on the model is better than "
                "acting on everyone or no-one, across the range of threshold probabilities a clinician might use. "
                "Net benefit = TP/n − FP/n × pₜ/(1 − pₜ).")
    if BINARY:
        target, s_dca, y_dca = CLASSES[POS], p[:, POS], y
    else:
        cc = st.selectbox("Target class (one-vs-rest)", CLASSES, index=0)
        ci = CLASSES.index(cc)
        target, s_dca, y_dca = cc, p[:, ci], (y == ci).astype(int)
    ts = np.linspace(0.01, 0.95, 95)
    nb, nb_all = net_benefit(y_dca, s_dca, ts)
    fig = go.Figure()
    fig.add_scatter(x=ts, y=nb, name="Model", line=dict(color=PALETTE[0], width=3))
    fig.add_scatter(x=ts, y=nb_all, name=f"Treat all as {target}", line=dict(color=PALETTE[1], dash="dash"))
    fig.add_scatter(x=ts, y=np.zeros_like(ts), name="Treat none", line=dict(color=INK2, dash="dot"))
    fig.update_yaxes(range=[min(-0.05, nb.min() - .02), max(y_dca.mean() * 1.1, nb.max() + .02)])
    fig.update_layout(title=f"Decision curve — {target}", xaxis_title="Threshold probability pₜ", yaxis_title="Net benefit")
    show(style_fig(fig, 440))
    useful = ts[(nb > nb_all) & (nb > 0)]
    if len(useful):
        finding(f"The model gives a higher net benefit than both default strategies for thresholds between "
                f"<b>{useful.min():.2f}</b> and <b>{useful.max():.2f}</b>.", "ok")
    sn = nb / max(y_dca.mean(), 1e-9)
    fig = go.Figure(go.Scatter(x=ts, y=sn, line=dict(color=PALETTE[2], width=2.5), name="standardised NB"))
    fig.update_layout(title="Standardised net benefit (fraction of maximum achievable)", xaxis_title="pₜ", yaxis_title="sNB")
    show(style_fig(fig, 320, legend=False))

# =========================================================================== 4. calibration
with tabs[3]:
    variants = {"Uncalibrated softmax": p_raw, f"Temperature-scaled (T={T:.2f})": softmax_t(p_raw, T),
                "MC-Dropout mean (30 passes)": A["mc_mean"].astype(np.float64)}
    rows, fig = [], go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(color=INK2, dash="dot"), name="perfect calibration")
    for i, (nm, pv) in enumerate(variants.items()):
        e, mc_, b, nl, bins = calib(pv, y)
        rows.append({"Probabilities": nm, "ECE ↓": round(e, 4), "MCE ↓": round(mc_, 4), "Brier ↓": round(b, 4),
                     "NLL ↓": round(nl, 4), "Accuracy": f"{(pv.argmax(1) == y).mean():.1%}"})
        fig.add_scatter(x=[r[0] for r in bins], y=[r[1] for r in bins], mode="lines+markers", name=nm,
                        line=dict(color=PALETTE[i]), marker=dict(size=[max(5, min(18, np.sqrt(r[2]))) for r in bins]))
    fig.update_layout(title="Reliability diagram (15 equal-width bins; marker size ∝ √n)",
                      xaxis_title="Mean predicted confidence", yaxis_title="Observed accuracy")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        show(style_fig(fig, 460))
    with c2:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        best = min(rows, key=lambda r: r["ECE ↓"])
        finding(f"Lowest expected calibration error: <b>{best['Probabilities']}</b> (ECE {best['ECE ↓']:.3f}). "
                "Temperature scaling (Guo et al., 2017) rescales logits with one scalar fitted on validation data.", "ok")
        fig = go.Figure()
        for i, (nm, m_) in enumerate([("correct", correct), ("wrong", ~correct)]):
            fig.add_histogram(x=p.max(1)[m_], name=nm, opacity=.75, nbinsx=25, marker_color=PALETTE[i * 3],
                              histnorm="probability")
        fig.update_layout(barmode="overlay", title="Confidence: correct vs wrong", xaxis_title="max p")
        show(style_fig(fig, 300))

# =========================================================================== 5. uncertainty
with tabs[4]:
    st.markdown("Predictive uncertainty is split into **aleatoric** (expected entropy, irreducible data noise) and "
                "**epistemic** (mutual information between prediction and weights, model ignorance) using 30 "
                "Monte-Carlo-Dropout passes (Gal & Ghahramani, 2016). Test-time augmentation (TTA) gives an "
                "input-perturbation view.")
    mc_correct = A["mc_mean"].argmax(1) == y
    unc = {"1 − max softmax (MSP)": 1 - p.max(1),
           "Softmax entropy": -(p * np.log(p + 1e-12)).sum(1),
           "MC total entropy": A["mc_total"], "MC aleatoric": A["mc_aleatoric"],
           "MC epistemic (MI)": A["mc_epistemic"], "TTA entropy": A["tta_total"]}
    err = (~correct).astype(int)
    rows = []
    for nm, u in unc.items():
        cov, risk, aurc = risk_cov(-u, correct)
        rows.append({"Uncertainty score": nm, "Error-detection AUROC ↑": round(roc_auc_score(err, u), 4),
                     "AURC ↓": round(aurc, 4)})
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Error-detection AUROC: how well the score separates misclassified from correct cases. "
                   "AURC: area under the risk–coverage curve (Geifman & El-Yaniv, 2017).")
        fig = go.Figure()
        for i, (nm, u) in enumerate([(k_, unc[k_]) for k_ in ["1 − max softmax (MSP)", "MC epistemic (MI)", "TTA entropy"]]):
            cov, risk, _ = risk_cov(-u, correct)
            fig.add_scatter(x=cov, y=risk, name=nm, line=dict(color=PALETTE[i]))
        fig.update_layout(title="Risk–coverage (selective prediction)", xaxis_title="Coverage (fraction auto-reported)",
                          yaxis_title="Error rate on covered cases")
        show(style_fig(fig, 380))
    with c2:
        fig = go.Figure()
        for i, (nm, m_) in enumerate([("correct", correct), ("misclassified", ~correct)]):
            fig.add_scatter(x=A["mc_aleatoric"][m_], y=A["mc_epistemic"][m_], mode="markers", name=nm,
                            marker=dict(color=PALETTE[i * 3], size=5 if nm == "correct" else 8, opacity=.6,
                                        symbol="circle" if nm == "correct" else "x"))
        fig.update_layout(title="Aleatoric vs epistemic uncertainty per test image",
                          xaxis_title="Aleatoric (nats)", yaxis_title="Epistemic MI (nats)")
        show(style_fig(fig, 420))
    st.markdown("##### Human–AI referral: send the most uncertain cases to a specialist")
    score = st.selectbox("Referral criterion", list(unc), index=0)
    refer = st.slider("Fraction of cases referred", 0.0, 0.5, 0.1, 0.01)
    u = unc[score]
    order = np.argsort(u)
    keep = order[: int(round(len(u) * (1 - refer)))]
    fr = np.linspace(0, 0.5, 51)
    acc_k, auc_k = [], []
    for f in fr:
        kk = order[: max(10, int(round(len(u) * (1 - f))))]
        acc_k.append(correct[kk].mean())
        try:
            auc_k.append(roc_auc_score(y[kk], p[kk, 1]) if BINARY else
                         roc_auc_score(y[kk], p[kk], multi_class="ovr", average="macro", labels=range(C)))
        except ValueError:
            auc_k.append(np.nan)
    k3 = st.columns(4)
    kpi(k3[0], "Auto-reported", f"{len(keep):,}", f"{1 - refer:.0%} of cases")
    kpi(k3[1], "Accuracy on retained", f"{correct[keep].mean():.1%}", f"vs {acc:.1%} on all")
    kpi(k3[2], "Errors caught by referral", f"{(~correct[order[len(keep):]]).sum()} / {(~correct).sum()}",
        "misclassifications sent to the specialist")
    kpi(k3[3], "Specialist workload", f"{len(u) - len(keep):,} cases", "per test set")
    fig = go.Figure()
    fig.add_scatter(x=fr, y=acc_k, name="Accuracy on retained", line=dict(color=PALETTE[0], width=2.5))
    fig.add_scatter(x=fr, y=auc_k, name="AUROC on retained", line=dict(color=PALETTE[2], width=2.5))
    fig.add_vline(x=refer, line_dash="dot", line_color=INK2)
    fig.update_layout(title="Performance of the automated arm vs referral rate", xaxis_title="Referral rate", yaxis_title="")
    show(style_fig(fig, 360))

# =========================================================================== 6. OOD
with tabs[5]:
    st.markdown("A deployed model will see images it was never trained for. Each score below ranks test images "
                "(in-distribution, positive) against an out-of-distribution set; higher = more in-distribution. "
                "Scores: maximum softmax probability (Hendrycks & Gimpel, 2017), energy (Liu et al., 2020), "
                "Mahalanobis distance in the embedding (Lee et al., 2018) and MC-Dropout epistemic uncertainty.")
    rows = []
    for ood_name, d in summary["ood"].items():
        for meth, (auroc, fpr95) in d.items():
            rows.append({"OOD set": ood_name, "Score": meth, "AUROC ↑": round(auroc, 4), "FPR@95%TPR ↓": round(fpr95, 4)})
    dfo = pd.DataFrame(rows)
    piv = dfo.pivot(index="OOD set", columns="Score", values="AUROC ↑")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        fig = px.imshow(piv, text_auto=".3f", color_continuous_scale="RdYlGn", zmin=0.5, zmax=1,
                        title="OOD detection AUROC (ID test set vs OOD set)", aspect="auto")
        show(style_fig(fig, 380, legend=False))
        st.dataframe(dfo, hide_index=True, use_container_width=True)
    with c2:
        o_set = st.selectbox("OOD set", list(summary["ood"]))
        o_meth = st.selectbox("Score", list(summary["ood"][o_set]), index=1)
        sid = A[f"ood_id__{o_meth}"]
        sod = A[f"ood__{o_set}__{o_meth}"]
        fig = go.Figure()
        fig.add_histogram(x=sid, name="in-distribution (test)", opacity=.7, marker_color=PALETTE[0], histnorm="probability", nbinsx=40)
        fig.add_histogram(x=sod, name=o_set, opacity=.7, marker_color=PALETTE[1], histnorm="probability", nbinsx=40)
        thr95 = np.percentile(sid, 5)
        fig.add_vline(x=thr95, line_dash="dot", line_color=INK2, annotation_text="95% TPR")
        fig.update_layout(barmode="overlay", title=f"{o_meth} distribution", xaxis_title="score (higher = ID)")
        show(style_fig(fig, 420))
        worst = dfo[dfo["AUROC ↑"] < 0.5]
        if len(worst):
            finding(f"{len(worst)} detector/OOD-set pairs score <b>below 0.5</b>: the model is <i>more</i> confident on "
                    "those foreign images than on real test images — the known over-confidence of softmax-based "
                    "scores far from the training data (Hein et al., CVPR 2019). Feature-space distances "
                    "(Mahalanobis) do not share this failure.", "warn")
        best = dfo.loc[dfo.groupby("OOD set")["AUROC ↑"].idxmax()]
        for _, r in best.iterrows():
            finding(f"<b>{r['OOD set']}</b>: best detector is {r['Score']} (AUROC {r['AUROC ↑']:.3f}).",
                    "ok" if r["AUROC ↑"] > .9 else "warn")

# =========================================================================== 7. robustness
with tabs[6]:
    rob = summary["robustness"]
    clean = rob["clean"]["acc"]
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for i, kind in enumerate([k_ for k_ in rob if k_ != "clean"]):
            fig.add_scatter(x=[0, 1, 2, 3, 4, 5], y=[clean] + [r["acc"] for r in rob[kind]], name=kind,
                            mode="lines+markers", line=dict(color=PALETTE[i % len(PALETTE)]))
        fig.update_layout(title=f"Accuracy under common corruptions (n = {meta['robustness_images']})",
                          xaxis_title="Severity (0 = clean)", yaxis_title="Accuracy")
        show(style_fig(fig, 420))
        mca = np.mean([r["acc"] for k_ in rob if k_ != "clean" for r in rob[k_]])
        finding(f"Mean corruption accuracy {mca:.1%} vs clean {clean:.1%} — relative robustness {mca / clean:.1%}. "
                "Corruptions mimic acquisition artefacts (quantum noise, defocus, exposure, compression).")
    with c2:
        adv = summary["fgsm"]
        fig = go.Figure()
        fig.add_scatter(x=[a["eps"] for a in adv], y=[a["acc"] for a in adv], name="accuracy", mode="lines+markers",
                        line=dict(color=PALETTE[1], width=2.5))
        fig.add_scatter(x=[a["eps"] for a in adv], y=[a["auc"] for a in adv], name="AUROC", mode="lines+markers",
                        line=dict(color=PALETTE[0], width=2.5))
        fig.update_layout(title="Adversarial sensitivity (FGSM, Goodfellow et al., 2015)",
                          xaxis_title="ε (intensity units out of 255)", yaxis_title="")
        show(style_fig(fig, 420))
        finding("Imperceptible perturbations of a few grey levels can flip predictions; this motivates adversarial "
                "training or certified defences before deployment.", "warn")
    heat = pd.DataFrame({k_: [r["auc"] for r in rob[k_]] for k_ in rob if k_ != "clean"}, index=[f"sev {s}" for s in range(1, 6)]).T
    fig = px.imshow(heat, text_auto=".3f", color_continuous_scale="RdYlGn", zmin=.5, zmax=1, aspect="auto",
                    title=f"AUROC by corruption and severity (clean AUROC {rob['clean']['auc']:.3f})")
    show(style_fig(fig, 340, legend=False))
    st.markdown("##### Hidden stratification by image-acquisition quality")
    st.caption("Test images are split into tertiles of a quality index; large gaps reveal shortcut learning or "
               "acquisition-dependent failure (Oakden-Rayner et al., 2020).")
    q = A["quality"]
    qnames = ["Mean intensity (exposure)", "RMS contrast", "Sharpness (Laplacian var.)", "Noise σ"]
    rows = []
    for qi, qn in enumerate(qnames):
        edges = np.percentile(q[:, qi], [33.3, 66.7])
        strata = np.digitize(q[:, qi], edges)
        for sidx, sname in enumerate(["low", "mid", "high"]):
            mm = strata == sidx
            lo_, hi_ = wilson(correct[mm].sum(), mm.sum())
            try:
                a_ = roc_auc_score(y[mm], p[mm, 1]) if BINARY else roc_auc_score(
                    y[mm], p[mm], multi_class="ovr", average="macro", labels=range(C))
            except ValueError:
                a_ = np.nan
            rows.append({"Quality index": qn, "Tertile": sname, "n": int(mm.sum()), "Accuracy": correct[mm].mean(),
                         "lo": lo_, "hi": hi_, "AUROC": a_})
    dq = pd.DataFrame(rows)
    fig = go.Figure()
    for i, sname in enumerate(["low", "mid", "high"]):
        d_ = dq[dq.Tertile == sname]
        fig.add_bar(x=d_["Quality index"], y=d_["Accuracy"], name=f"{sname} tertile", marker_color=PALETTE[i],
                    error_y=dict(type="data", symmetric=False, array=d_["hi"] - d_["Accuracy"], arrayminus=d_["Accuracy"] - d_["lo"]))
    fig.update_layout(barmode="group", title="Accuracy by quality tertile (Wilson 95% CI)", yaxis_title="Accuracy",
                      yaxis_range=[max(0, dq["lo"].min() - .05), 1])
    show(style_fig(fig, 380))
    gap = dq.groupby("Quality index")["Accuracy"].agg(lambda v: v.max() - v.min()).sort_values(ascending=False)
    finding(f"Largest performance gap across strata: <b>{gap.index[0]}</b> ({gap.iloc[0]:.1%} between tertiles).",
            "warn" if gap.iloc[0] > .05 else "ok")

# =========================================================================== 8. representations
with tabs[7]:
    ti = A["tsne_idx"]
    emb = A["tsne"]
    view = st.radio("Projection", ["t-SNE", "PCA"], horizontal=True)
    coords, yy, cc_ = (emb, y[ti], correct[ti]) if view == "t-SNE" else (A["pca"], y, correct)
    fig = go.Figure()
    for c in range(C):
        m_ = yy == c
        fig.add_scatter(x=coords[m_ & cc_, 0], y=coords[m_ & cc_, 1], mode="markers", name=CLASSES[c],
                        marker=dict(color=PALETTE[c % len(PALETTE)], size=5, opacity=.65))
        fig.add_scatter(x=coords[m_ & ~cc_, 0], y=coords[m_ & ~cc_, 1], mode="markers", showlegend=False,
                        marker=dict(color=PALETTE[c % len(PALETTE)], size=10, symbol="x", line=dict(width=1, color=INK)))
    fig.update_layout(title=f"{view} of the 128-d embedding layer (× = misclassified)", xaxis_title="dim 1", yaxis_title="dim 2")
    show(style_fig(fig, 560))
    finding(f"Silhouette coefficient of true classes in the embedding: <b>{meta['silhouette_embedding']:.3f}</b> "
            "(−1 to 1; higher = better-separated clusters). Misclassified cases sit on cluster boundaries, "
            "where epistemic uncertainty is also highest.")

# =========================================================================== 9. XAI (live)
with tabs[8]:
    gs = summary["faithfulness"]
    fracs = summary["faithfulness_fractions"]
    st.markdown("##### Population-level faithfulness of explanation methods")
    st.caption(f"Deletion / insertion metrics (Petsiuk et al., 2018) on {meta['faithfulness_images']} test images: "
               "pixels are removed (replaced by a blurred copy) or restored in order of attributed importance. "
               "Faithful maps make the class probability fall fast on deletion (low AUC) and rise fast on insertion "
               "(high AUC). A random map is the null baseline.")
    dff = pd.DataFrame([{"Method": m_, "Deletion AUC ↓": f"{v['deletion_auc']:.3f} ± {v['deletion_sd']:.3f}",
                         "Insertion AUC ↑": f"{v['insertion_auc']:.3f} ± {v['insertion_sd']:.3f}",
                         "Ins − Del ↑": round(v["insertion_auc"] - v["deletion_auc"], 3)} for m_, v in gs.items()])
    c1, c2, c3 = st.columns([1.1, 1, 1])
    with c1:
        st.dataframe(dff.sort_values("Ins − Del ↑", ascending=False), hide_index=True, use_container_width=True)
        rnd = np.array(gs["Random"]["ins_all"]) - np.array(gs["Random"]["del_all"])
        from scipy.stats import wilcoxon
        lines = []
        for m_, v in gs.items():
            if m_ == "Random":
                continue
            diff = np.array(v["ins_all"]) - np.array(v["del_all"])
            try:
                pw = wilcoxon(diff, rnd).pvalue
            except ValueError:
                pw = np.nan
            lines.append(f"{m_}: p = {pw:.2g}")
        st.caption("Paired Wilcoxon signed-rank test of (Ins − Del) vs the random map: " + "; ".join(lines))
    for col, kname, ttl in [(c2, "del_curve", "Deletion curves"), (c3, "ins_curve", "Insertion curves")]:
        with col:
            fig = go.Figure()
            for m_, v in gs.items():
                fig.add_scatter(x=fracs, y=v[kname], name=m_, line=dict(color=METHOD_COLOR.get(m_, INK2),
                                                                        dash="dot" if m_ == "Random" else "solid"))
            fig.update_layout(title=ttl, xaxis_title="Fraction of pixels", yaxis_title="P(predicted class)")
            show(style_fig(fig, 400))

    st.markdown("---")
    st.markdown("##### Case-level explanation (live)")
    if case_img is None:
        st.info("Choose a gallery image or upload one in the sidebar.")
    else:
        import xai_utils as X
        model, ex = load_model(key)

        @st.cache_data(show_spinner="Computing attribution maps…")
        def case_analysis(_ex, img_bytes, shape, k_):
            img = np.frombuffer(img_bytes, np.float32).reshape(shape)
            maps, cls = X.all_maps(_ex, img)
            probs = _ex.predict(img[None])[0]
            mc = X.mc_dropout(_ex, img[None], T=50)[:, 0]
            di = {m_: X.deletion_insertion(_ex, img, mp, cls) for m_, mp in maps.items()}
            return maps, cls, probs, mc, di

        maps, cls, probs, mc, di = case_analysis(ex, case_img.astype(np.float32).tobytes(), case_img.shape, key)
        pc = softmax_t(probs[None], T)[0] if use_ts else probs
        _, tot, ale, epi = X.decompose(mc[:, None, :])
        c1, c2 = st.columns([1, 1.6])
        with c1:
            st.image(big(to01_u8(case_img), 240), caption="Input")
            verdict = f"**Prediction: {CLASSES[pc.argmax()]}** ({pc.max():.1%})"
            if case_label is not None:
                verdict += " ✅" if pc.argmax() == case_label else f" ❌ (truth: {CLASSES[case_label]})"
            st.markdown(verdict)
            st.markdown(f"Total uncertainty {tot[0]:.3f} nats · aleatoric {ale[0]:.3f} · epistemic {epi[0]:.4f}")
            if epi[0] > np.percentile(A["mc_epistemic"], 95):
                finding("Epistemic uncertainty is above the 95th percentile of the test set — refer for expert review.", "warn")
        with c2:
            fig = go.Figure()
            fig.add_box(x=[CLASSES[c] for c in range(C) for _ in range(len(mc))],
                        y=np.concatenate([mc[:, c] for c in range(C)]), name="MC-Dropout samples (50)",
                        marker_color=PALETTE[0], boxpoints="all", jitter=.4, pointpos=0, marker_size=3)
            fig.add_scatter(x=CLASSES, y=pc, mode="markers", name="deterministic prediction",
                            marker=dict(symbol="diamond", size=12, color=PALETTE[1]))
            fig.update_layout(title="Predictive distribution", yaxis_title="Probability", yaxis_range=[0, 1])
            show(style_fig(fig, 340))
        cols = st.columns(len(maps))
        for col, (m_, mp) in zip(cols, maps.items()):
            col.image(Image.fromarray(to01_u8(X.overlay(case_img, mp))).resize((180, 180), Image.BILINEAR),
                      caption=f"{m_}\nDel {di[m_][3]:.2f} · Ins {di[m_][4]:.2f}")
        cols = st.columns(len(maps))
        for col, (m_, mp) in zip(cols, maps.items()):
            col.image(Image.fromarray(to01_u8(X.jet(mp))).resize((180, 180), Image.NEAREST), caption="raw map")
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            for m_, (fr_, dc, ic, da, ia) in di.items():
                fig.add_scatter(x=fr_, y=dc, name=m_, line=dict(color=METHOD_COLOR[m_]))
            fig.update_layout(title="Deletion curves — this image", xaxis_title="Fraction deleted", yaxis_title="P(class)")
            show(style_fig(fig, 380))
        with c2:
            names = list(maps)
            R = np.array([[spearmanr(maps[a].ravel(), maps[b].ravel())[0] for b in names] for a in names])
            fig = px.imshow(R, x=names, y=names, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu",
                            title="Agreement between methods (Spearman ρ)")
            show(style_fig(fig, 380, legend=False))

        with st.expander("🧪 Sanity check — cascading model-parameter randomisation (Adebayo et al., 2018)"):
            st.caption("If an explanation does not change when the trained weights are replaced by random ones, it "
                       "reflects the image rather than the model. Layers are randomised from the top down.")

            @st.cache_data(show_spinner="Randomising layers…")
            def sanity(_model, img_bytes, shape, k_):
                import tensorflow as tf
                img = np.frombuffer(img_bytes, np.float32).reshape(shape)
                base = X.explainable_from_sequential(_model, "last_conv")
                ref_cam, c0 = X.grad_cam(base, img)
                ref_sal = X.saliency(base, img, c0)
                clone = tf.keras.models.clone_model(_model)
                clone.set_weights(_model.get_weights())
                fresh = tf.keras.models.clone_model(_model)  # new random initialisation
                out = []
                for lname in ["probs", "embedding", "last_conv", "conv3", "conv2", "conv1"]:
                    clone.get_layer(lname).set_weights(fresh.get_layer(lname).get_weights())
                    ex_r = X.explainable_from_sequential(clone, "last_conv")
                    cam_r, _ = X.grad_cam(ex_r, img, c0)
                    sal_r = X.saliency(ex_r, img, c0)
                    out.append((lname, spearmanr(ref_cam.ravel(), cam_r.ravel())[0],
                                spearmanr(ref_sal.ravel(), sal_r.ravel())[0], cam_r, sal_r))
                return out

            res = sanity(model, case_img.astype(np.float32).tobytes(), case_img.shape, key)
            fig = go.Figure()
            fig.add_scatter(x=[r[0] for r in res], y=[r[1] for r in res], name="Grad-CAM", mode="lines+markers",
                            line=dict(color=METHOD_COLOR["Grad-CAM"]))
            fig.add_scatter(x=[r[0] for r in res], y=[r[2] for r in res], name="Saliency", mode="lines+markers",
                            line=dict(color=METHOD_COLOR["Saliency"]))
            fig.update_layout(title="Rank correlation with the trained-model map (lower = passes the test)",
                              xaxis_title="Randomised down to layer", yaxis_title="Spearman ρ", yaxis_range=[-1, 1])
            show(style_fig(fig, 340))
            cols = st.columns(len(res))
            for col, r in zip(cols, res):
                col.image(Image.fromarray(to01_u8(X.overlay(case_img, np.nan_to_num(r[3])))).resize((130, 130)),
                          caption=f"≤ {r[0]}")

# =========================================================================== 10. radiomics
with tabs[9]:
    if case_img is None:
        st.info("Choose a gallery image or upload one in the sidebar.")
    else:
        import xai_utils as X
        g = X.to_gray(case_img).astype(np.float64)
        fo = X.first_order(g)
        gl, P = X.glcm_features(g)
        qi = X.quality_indices(g)
        c1, c2, c3 = st.columns(3)
        c1.image(big(to01_u8(g[..., None]), 220), caption="Grey-level view")
        c2.image(big(to01_u8(X.equalize(g)[..., None]), 220), caption="Histogram-equalised")
        edge = np.hypot(*np.gradient(g))
        c3.image(big(to01_u8((edge / (edge.max() + 1e-9))[..., None]), 220), caption="Gradient magnitude")
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Histogram(x=g.ravel(), nbinsx=64, marker_color=PALETTE[0]))
            fig.update_layout(title="Intensity histogram", xaxis_title="Intensity [0, 1]", yaxis_title="pixels")
            show(style_fig(fig, 320, legend=False))
            fig = px.imshow(P, color_continuous_scale="Viridis", title="Grey-level co-occurrence matrix (32 levels)")
            show(style_fig(fig, 360, legend=False))
        with c2:
            feat = {**{f"First-order · {k_}": v for k_, v in fo.items()}, **{f"Texture · {k_}": v for k_, v in gl.items()},
                    **{f"Quality · {k_}": v for k_, v in qi.items()}}
            st.dataframe(pd.DataFrame({"Feature": list(feat), "Value": [f"{v:.4g}" for v in feat.values()]}),
                         hide_index=True, use_container_width=True, height=560)
            st.caption("Feature definitions follow the Image Biomarker Standardisation Initiative (Zwanenburg et al., "
                       "2020) where applicable; whole-image ROI, 32-level quantisation, offsets 1–2 px at 0/45/90/135°.")
        st.markdown("##### Dataset-shift check for this image")
        q = A["quality"]
        mine = [g.mean(), qi["RMS contrast"], qi["Sharpness (Laplacian var.)"], qi["Noise σ (Immerkær)"]]
        names = ["Mean intensity", "RMS contrast", "Sharpness", "Noise σ"]
        pct = [float((q[:, i] < v).mean() * 100) for i, v in enumerate(mine)]
        fig = go.Figure(go.Bar(x=names, y=pct, marker_color=[PALETTE[1] if (v < 2.5 or v > 97.5) else PALETTE[2] for v in pct],
                               text=[f"{v:.0f}th" for v in pct], textposition="outside"))
        fig.add_hrect(y0=2.5, y1=97.5, fillcolor=PALETTE[2], opacity=.08, line_width=0)
        fig.update_layout(title="Percentile of this image within the test-set distribution (shaded = central 95%)",
                          yaxis_range=[0, 110], yaxis_title="percentile")
        show(style_fig(fig, 330, legend=False))
        model, ex = load_model(key)
        import tensorflow as tf
        layers_ = model.layers[: [l.name for l in model.layers].index("embedding") + 1]
        e = X._run_layers(layers_, tf.convert_to_tensor(case_img[None], tf.float32)).numpy()
        md = X.mahalanobis_score(e, A["maha_mus"], A["maha_prec"])[0]
        W_, b_ = model.get_layer("probs").get_weights()
        en = X.energy_score(e @ W_ + b_)[0]
        thr_md = np.percentile(A["ood_id__Mahalanobis"], 5)
        thr_en = np.percentile(A["ood_id__Energy"], 5)
        flags = [md < thr_md, en < thr_en] + [(v < 2.5 or v > 97.5) for v in pct]
        k4 = st.columns(3)
        kpi(k4[0], "Mahalanobis score", f"{md:.1f}", f"ID threshold (95% TPR) {thr_md:.1f}")
        kpi(k4[1], "Energy score", f"{en:.2f}", f"ID threshold (95% TPR) {thr_en:.2f}")
        kpi(k4[2], "Shift flags", f"{sum(flags)} / {len(flags)}", "OOD scores + quality percentiles")
        if md < thr_md or en < thr_en:
            finding("This image looks <b>out-of-distribution</b> for the model; its prediction should not be trusted.", "warn")
        else:
            finding("This image is consistent with the training distribution on both OOD scores.", "ok")

# =========================================================================== 11. model card
with tabs[10]:
    e_raw = calib(p_raw, y)[0]
    e_ts = calib(softmax_t(p_raw, T), y)[0]
    best_ood = {k_: max(v.values(), key=lambda t_: t_[0])[0] for k_, v in summary["ood"].items()}
    ood_txt = "; ".join(f"{k_.split(' (')[0]} {v:.3f}" for k_, v in best_ood.items())
    card = f"""
**Model card — {meta['dataset']} classifier** (Mitchell et al., 2019; reporting items follow TRIPOD+AI 2024 and CLAIM 2024)

| Item | Detail |
|---|---|
| Intended use | Research and teaching demonstration of trustworthy-AI evaluation. **Not** for diagnosis. |
| Data | {meta['source']}. Train/val/test {meta['n_train']:,} / {meta['n_val']:,} / {meta['n_test']:,}; 64×64 px. |
| Outcome | {', '.join(CLASSES)} (reference standard: labels released with MedMNIST v2). |
| Model | {meta['architecture']}; {meta['params']:,} parameters; TF {meta['tensorflow']}. |
| Training | {meta['training']}; seed {meta['seed']}; best epoch {meta['best_epoch']}. |
| Discrimination | Accuracy {acc:.1%}; {'AUROC' if BINARY else 'macro AUROC'} {macro_auc:.3f}. |
| Calibration | ECE {e_raw:.3f} → {e_ts:.3f} after temperature scaling (T = {T:.2f}). |
| Uncertainty | 30-pass MC-Dropout; aleatoric/epistemic decomposition; referral analysis provided. |
| Robustness | Clean {summary['robustness']['clean']['acc']:.1%}; 6 corruptions × 5 severities; FGSM ε ≤ 8/255. |
| OOD detection | Best AUROC per set: {ood_txt}. |
| Explainability | Grad-CAM, Grad-CAM++, saliency, SmoothGrad, IG, occlusion; deletion/insertion faithfulness; parameter-randomisation sanity check. |
| Limitations | Single-centre public data; 64 px resolution; no external validation; no demographic metadata, so fairness across patient groups cannot be assessed — quality-stratified performance is reported instead. |
"""
    st.markdown(card)
    out = pd.DataFrame({"test_index": np.arange(len(y)), "y_true": [CLASSES[i] for i in y],
                        "y_pred": [CLASSES[i] for i in pred], "correct": correct,
                        **{f"p_{c}": p[:, i] for i, c in enumerate(CLASSES)},
                        "mc_total": A["mc_total"], "mc_aleatoric": A["mc_aleatoric"], "mc_epistemic": A["mc_epistemic"]})
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Per-image predictions + uncertainty (CSV)", out.to_csv(index=False).encode(),
                       f"{key}_predictions.csv", "text/csv", use_container_width=True)
    c2.download_button("⬇️ Model card (Markdown)", card.encode(), f"{key}_model_card.md", "text/markdown",
                       use_container_width=True)

st.markdown("---")
st.markdown("<div class='small-note' style='text-align:center'>Data: MedMNIST v2 (Yang et al., Scientific Data 2023), "
            "CC BY 4.0. Pre-computed with <code>benchmark/train_medical.py</code>.</div>", unsafe_allow_html=True)
