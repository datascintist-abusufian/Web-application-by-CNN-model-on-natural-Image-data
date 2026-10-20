"""
Model Comparison dashboard — a separate Streamlit page (auto-discovered from /pages).

Reads the artefacts written by benchmark/train_and_benchmark.py:
    results/benchmark_summary.json   learning curves, latency, robustness, metadata
    results/test_predictions.npz     y_true + softmax probabilities per model
    results/deployment_audit.json    (optional) audit of the models served by app.py

Every metric shown here is computed from those files at page load — nothing is
hard-coded. TensorFlow is NOT imported, so the page opens instantly.
"""
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import binomtest
from sklearn.metrics import (auc, confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)

st.set_page_config(page_title="Model Comparison · CIFAR-10", page_icon="📊", layout="wide")

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MODEL_ORDER = ["CNN", "ResNet50", "VGG16", "MobileNetV2"]
# Fixed categorical order (validated light-mode palette) + a distinct marker per model,
# so identity never relies on colour alone.
MODEL_COLOR = {"CNN": "#2a78d6", "ResNet50": "#eb6834", "VGG16": "#1baf7a", "MobileNetV2": "#eda100"}
MODEL_SYMBOL = {"CNN": "circle", "ResNet50": "square", "VGG16": "diamond", "MobileNetV2": "triangle-up"}
MODEL_DASH = {"CNN": "solid", "ResNet50": "solid", "VGG16": "solid", "MobileNetV2": "solid"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"

st.markdown("""
<style>
.gradient-header{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;
  padding:1.6rem 2rem;border-radius:15px;margin:.5rem 0 1rem;text-align:center}
.gradient-header h1{color:#fff;margin:0;font-size:2rem}
.gradient-header p{margin:.35rem 0 0;opacity:.9}
.kpi{background:#f8f9fa;border-radius:12px;padding:1rem 1.1rem;border-left:5px solid #667eea;height:100%}
.kpi .lbl{font-size:.78rem;color:#52514e;text-transform:uppercase;letter-spacing:.04em}
.kpi .val{font-size:1.55rem;font-weight:700;color:#0b0b0b;line-height:1.2;margin-top:.2rem}
.kpi .sub{font-size:.82rem;color:#52514e}
.finding{background:#f8f9fa;border-radius:10px;padding:.75rem 1rem;margin:.35rem 0;border-left:4px solid #17a2b8}
.finding.warn{border-left-color:#dc3545;background:#fff5f5}
.small-note{font-size:.82rem;color:#52514e}
</style>
""", unsafe_allow_html=True)


# =========================================================================== helpers
def style_fig(fig, height=420, legend=True):
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=10, r=10, t=95 if legend else 55, b=10),
        font=dict(color=INK, size=13), hovermode=fig.layout.hovermode or "closest", showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, title=None,
                    font=dict(color=INK2, size=12)),
    )
    if fig.layout.title.text:
        fig.update_layout(title=dict(y=0.98, yanchor="top", x=0, xanchor="left", font=dict(size=16)))
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, title_font=dict(color=INK2))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, title_font=dict(color=INK2))
    return fig


def show(fig):
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


@st.cache_data(show_spinner=False)
def load_results(summary_mtime, preds_mtime):
    with open(RESULTS / "benchmark_summary.json") as f:
        summary = json.load(f)
    npz = np.load(RESULTS / "test_predictions.npz")
    y = npz["y_true"].astype(int)
    probs = {m: npz[m].astype(np.float64) for m in npz.files if m != "y_true"}
    for m in probs:  # float16 storage → renormalise
        probs[m] = probs[m] / probs[m].sum(1, keepdims=True)
    audit = None
    if (RESULTS / "deployment_audit.json").exists():
        with open(RESULTS / "deployment_audit.json") as f:
            audit = json.load(f)
    digest = hashlib.md5(npz["y_true"].tobytes() + b"".join(npz[m].tobytes() for m in sorted(probs))).hexdigest()
    return summary, y, probs, audit, digest


def ece_score(p, y, n_bins=15):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            acc_b, conf_b = (pred[mask] == y[mask]).mean(), conf[mask].mean()
            ece += mask.mean() * abs(acc_b - conf_b)
            rows.append((conf_b, acc_b, int(mask.sum())))
    return ece, rows


@st.cache_data(show_spinner=False)
def bootstrap_ci(correct, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(correct)
    stats = correct[rng.integers(0, n, (n_boot, n))].mean(1)
    return np.percentile(stats, [2.5, 97.5])


def holm(pvals):
    order = np.argsort(pvals)
    m, adj, running = len(pvals), np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


@st.cache_data(show_spinner=False)
def compute_metrics(_y, _probs, models, n_classes, data_digest):
    y = _y
    rows, per_class = [], {}
    onehot = np.eye(n_classes)[y]
    for m in models:
        p = _probs[m]
        pred = p.argmax(1)
        correct = (pred == y).astype(float)
        lo, hi = bootstrap_ci(correct)
        P, R, F, _ = precision_recall_fscore_support(y, pred, labels=range(n_classes), zero_division=0)
        top3 = np.mean([y[i] in np.argsort(p[i])[-3:] for i in range(len(y))])
        ece, _ = ece_score(p, y)
        nll = -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)))
        brier = np.mean(np.sum((p - onehot) ** 2, 1))
        try:
            mauc = roc_auc_score(y, p, multi_class="ovr", average="macro")
        except ValueError:
            mauc = np.nan
        rows.append(dict(Model=m, Accuracy=correct.mean(), CI_lo=lo, CI_hi=hi,
                         **{"Macro precision": P.mean(), "Macro recall": R.mean(), "Macro F1": F.mean(),
                            "Top-3 accuracy": top3, "ECE": ece, "NLL": nll, "Brier": brier,
                            "Macro AUC": mauc}))
        per_class[m] = dict(precision=P, recall=R, f1=F)
    return pd.DataFrame(rows).set_index("Model"), per_class


# =========================================================================== header
st.markdown("""
<div class="gradient-header">
  <h1>📊 Model Comparison Dashboard</h1>
  <p>CNN vs ResNet50 vs VGG16 vs MobileNetV2 on CIFAR-10 — accuracy, learning curves, calibration,
  robustness, efficiency and statistical significance</p>
</div>""", unsafe_allow_html=True)

if not (RESULTS / "benchmark_summary.json").exists() or not (RESULTS / "test_predictions.npz").exists():
    st.warning("No benchmark results found in `results/`.")
    st.markdown("Generate them once (CPU ≈ 30 min, GPU a few minutes), commit the `results/` folder, "
                "and this page will populate automatically:")
    st.code("python benchmark/train_and_benchmark.py", language="bash")
    st.stop()

summary, y_true, probs, audit, DIGEST = load_results(
    (RESULTS / "benchmark_summary.json").stat().st_mtime, (RESULTS / "test_predictions.npz").stat().st_mtime)
meta = summary["meta"]
CLASSES = meta["class_names"]
NC = len(CLASSES)
models = [m for m in MODEL_ORDER if m in probs and m in summary["models"]]
if len(models) < 2:
    st.info(f"Only {len(models)} model(s) in the results so far — rerun the benchmark to compare all four.")
info = summary["models"]
metrics, per_class = compute_metrics(y_true, probs, tuple(models), NC, DIGEST)

# ------------------------------------------------------------------ sidebar controls
with st.sidebar:
    st.header("📊 Comparison controls")
    selected = st.multiselect("Models to display", models, default=models,
                              help="Colours stay fixed to each model when you filter.")
    if not selected:
        selected = models
    st.markdown("---")
    st.markdown(f"""<div class="small-note">
    <b>Protocol</b><br>
    Train / val / test: {meta['n_train']:,} / {meta['n_val']:,} / {meta['n_test']:,} (stratified, seed {meta['seed']})<br>
    Transfer input: {meta['transfer_input_size']}px · CNN input: {meta['cnn_input_size']}px<br>
    {meta['optimizer']} · batch {meta['batch_size']} · ≤{meta['max_epochs']} epochs<br>
    Early stopping: {meta['early_stopping']}<br>
    Hardware: {meta['hardware']}<br>
    Generated: {meta['created_utc']}</div>""", unsafe_allow_html=True)

# =========================================================================== KPI row
best_acc = metrics.loc[selected, "Accuracy"].idxmax()
best_f1 = metrics.loc[selected, "Macro F1"].idxmax()
fastest = min(selected, key=lambda m: info[m]["latency_ms_bs1"])
best_cal = metrics.loc[selected, "ECE"].idxmin()
k = st.columns(4)
kpis = [
    ("Best accuracy", best_acc, f"{metrics.loc[best_acc, 'Accuracy']:.1%}",
     f"95% CI {metrics.loc[best_acc, 'CI_lo']:.1%} – {metrics.loc[best_acc, 'CI_hi']:.1%}"),
    ("Best macro-F1", best_f1, f"{metrics.loc[best_f1, 'Macro F1']:.3f}", "balanced across 10 classes"),
    ("Fastest inference", fastest, f"{info[fastest]['latency_ms_bs1']:.1f} ms",
     "per image, batch = 1, CPU"),
    ("Best calibrated", best_cal, f"ECE {metrics.loc[best_cal, 'ECE']:.3f}", "lower is better"),
]
for col, (lbl, m, val, sub) in zip(k, kpis):
    col.markdown(f"""<div class="kpi" style="border-left-color:{MODEL_COLOR[m]}">
      <div class="lbl">{lbl}</div><div class="val">{val}</div>
      <div class="sub"><b>{m}</b> · {sub}</div></div>""", unsafe_allow_html=True)

st.write("")
tab_overview, tab_curves, tab_class, tab_cal, tab_roc, tab_rob, tab_eff, tab_stats, tab_audit = st.tabs([
    "🏁 Comparison", "📈 Learning curves", "🧩 Per-class", "🎯 Calibration", "📉 ROC",
    "🛡️ Robustness", "⚡ Efficiency", "🧪 Significance & ensemble", "🔎 Deployment audit"])

# =========================================================================== 1. comparison
with tab_overview:
    st.subheader("Head-to-head comparison")
    table = pd.DataFrame({
        "Pretraining": [info[m]["pretrained"] for m in selected],
        "Input (px)": [info[m]["input_size"] for m in selected],
        "Total params (M)": [info[m]["total_params"] / 1e6 for m in selected],
        "Trainable params (M)": [info[m]["trainable_params"] / 1e6 for m in selected],
        "Accuracy": metrics.loc[selected, "Accuracy"].values,
        "95% CI": [f"{metrics.loc[m, 'CI_lo']:.1%} – {metrics.loc[m, 'CI_hi']:.1%}" for m in selected],
        "Macro F1": metrics.loc[selected, "Macro F1"].values,
        "Macro precision": metrics.loc[selected, "Macro precision"].values,
        "Macro recall": metrics.loc[selected, "Macro recall"].values,
        "Top-3 accuracy": metrics.loc[selected, "Top-3 accuracy"].values,
        "Macro AUC": metrics.loc[selected, "Macro AUC"].values,
        "ECE": metrics.loc[selected, "ECE"].values,
        "NLL": metrics.loc[selected, "NLL"].values,
        "Latency bs=1 (ms)": [info[m]["latency_ms_bs1"] for m in selected],
        "Best epoch": [f"{info[m]['best_epoch']} / {info[m]['epochs_run']}" for m in selected],
        "Training time (s)": [info[m]["fit_time_s"] + info[m].get("feature_time_s", 0) for m in selected],
    }, index=pd.Index(selected, name="Model"))
    pct = st.column_config.NumberColumn(format="%.3f")
    st.dataframe(table, use_container_width=True, column_config={
        "Accuracy": st.column_config.ProgressColumn("Accuracy", format="%.3f", min_value=0, max_value=1),
        "Macro F1": pct, "Macro precision": pct, "Macro recall": pct, "Top-3 accuracy": pct,
        "Macro AUC": pct, "ECE": pct, "NLL": pct,
        "Total params (M)": st.column_config.NumberColumn(format="%.2f"),
        "Trainable params (M)": st.column_config.NumberColumn(format="%.2f"),
        "Training time (s)": st.column_config.NumberColumn(format="%.0f"),
    })
    st.download_button("⬇️ Download comparison (CSV)", table.to_csv().encode(), "model_comparison.csv",
                       "text/csv")

    c1, c2 = st.columns([3, 2])
    with c1:
        bar_metrics = ["Accuracy", "Macro F1", "Top-3 accuracy", "Macro AUC"]
        fig = go.Figure()
        for m in selected:
            vals = [metrics.loc[m, b] for b in bar_metrics]
            err = None
            fig.add_bar(name=m, x=bar_metrics, y=vals, marker_color=MODEL_COLOR[m],
                        marker_line=dict(color="white", width=2), error_y=err,
                        hovertemplate=f"<b>{m}</b><br>%{{x}}: %{{y:.3f}}<extra></extra>")
        fig.update_layout(barmode="group", title="Headline metrics (test set)", yaxis_range=[0, 1.02],
                          bargap=0.25, bargroupgap=0.08)
        show(style_fig(fig, 400))
    with c2:
        fig = go.Figure()
        for m in selected:
            lo, hi, a = metrics.loc[m, "CI_lo"], metrics.loc[m, "CI_hi"], metrics.loc[m, "Accuracy"]
            fig.add_scatter(x=[a], y=[m], mode="markers", name=m,
                            marker=dict(color=MODEL_COLOR[m], size=13, symbol=MODEL_SYMBOL[m],
                                        line=dict(color="white", width=2)),
                            error_x=dict(type="data", symmetric=False, array=[hi - a], arrayminus=[a - lo],
                                         color=MODEL_COLOR[m], thickness=2, width=6),
                            hovertemplate=f"<b>{m}</b><br>acc {a:.3f}<br>95% CI {lo:.3f}–{hi:.3f}<extra></extra>")
        fig.update_layout(title="Accuracy with 95% bootstrap CI", xaxis_title="Test accuracy",
                          yaxis=dict(categoryorder="array", categoryarray=selected[::-1]))
        show(style_fig(fig, 400, legend=False))

    # ---- auto-generated findings
    st.subheader("Key findings")
    ranked = metrics.loc[selected].sort_values("Accuracy", ascending=False)
    top, second = ranked.index[0], ranked.index[1] if len(ranked) > 1 else None
    findings = [f"<b>{top}</b> leads with {ranked.iloc[0]['Accuracy']:.1%} test accuracy"
                + (f", {ranked.iloc[0]['Accuracy'] - ranked.iloc[1]['Accuracy']:+.1%} over {second}." if second else ".")]
    gaps = {m: info[m]["history"]["accuracy"][info[m]["best_epoch"] - 1]
            - info[m]["history"]["val_accuracy"][info[m]["best_epoch"] - 1] for m in selected}
    worst_gap = max(gaps, key=gaps.get)
    findings.append(f"Largest train–validation gap at the selected epoch: <b>{worst_gap}</b> "
                    f"({gaps[worst_gap]:+.1%}) — the clearest overfitting signal.")
    acc_per_ms = {m: metrics.loc[m, 'Accuracy'] / info[m]['latency_ms_bs1'] for m in selected}
    eff = max(acc_per_ms, key=acc_per_ms.get)
    findings.append(f"Best accuracy-per-millisecond trade-off: <b>{eff}</b> "
                    f"({metrics.loc[eff, 'Accuracy']:.1%} at {info[eff]['latency_ms_bs1']:.1f} ms/image).")
    hardest = CLASSES[int(np.argmin(np.mean([per_class[m]['f1'] for m in selected], 0)))]
    easiest = CLASSES[int(np.argmax(np.mean([per_class[m]['f1'] for m in selected], 0)))]
    findings.append(f"Hardest class across models: <b>{hardest}</b>; easiest: <b>{easiest}</b>.")
    for f_ in findings:
        st.markdown(f'<div class="finding">{f_}</div>', unsafe_allow_html=True)

# =========================================================================== 2. learning curves
with tab_curves:
    st.subheader("Learning curves — all models on one timeline")
    show_train = st.toggle("Show training curves (dashed)", value=True)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy per epoch", "Loss per epoch"),
                        horizontal_spacing=0.08)
    for m in selected:
        h = info[m]["history"]
        ep = list(range(1, len(h["loss"]) + 1))
        be = info[m]["best_epoch"]
        for col, key in ((1, "accuracy"), (2, "loss")):
            fig.add_scatter(x=ep, y=h[f"val_{key}"], mode="lines", name=m, legendgroup=m,
                            showlegend=(col == 1), line=dict(color=MODEL_COLOR[m], width=2.2),
                            hovertemplate=f"<b>{m}</b> val {key}<br>epoch %{{x}}: %{{y:.4f}}<extra></extra>",
                            row=1, col=col)
            if show_train:
                fig.add_scatter(x=ep, y=h[key], mode="lines", name=f"{m} (train)", legendgroup=m,
                                showlegend=False, line=dict(color=MODEL_COLOR[m], width=1.6, dash="dash"),
                                opacity=0.75,
                                hovertemplate=f"<b>{m}</b> train {key}<br>epoch %{{x}}: %{{y:.4f}}<extra></extra>",
                                row=1, col=col)
            fig.add_scatter(x=[be], y=[h[f"val_{key}"][be - 1]], mode="markers", legendgroup=m,
                            showlegend=False,
                            marker=dict(color=MODEL_COLOR[m], size=11, symbol=MODEL_SYMBOL[m],
                                        line=dict(color="white", width=2)),
                            hovertemplate=f"<b>{m}</b> best epoch {be}<br>val {key} %{{y:.4f}}<extra></extra>",
                            row=1, col=col)
    max_ep = max(len(info[m]["history"]["loss"]) for m in selected)
    fig.update_xaxes(title_text="Epoch", tickformat="d", dtick=1 if max_ep <= 12 else None)
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="Cross-entropy loss", row=1, col=2)
    fig.update_layout(hovermode="x unified")
    fig.update_annotations(font=dict(size=15, color=INK), yshift=6)
    style_fig(fig, 500)
    fig.update_layout(margin=dict(t=50, b=90), legend=dict(yanchor="top", y=-0.16, x=0.5, xanchor="center"))
    show(fig)
    st.caption("Solid = validation, dashed = training. Marker = epoch whose weights were kept "
               "(lowest validation loss, early-stopping patience 8).")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for m in selected:
            h = info[m]["history"]
            gap = np.array(h["accuracy"]) - np.array(h["val_accuracy"])
            fig.add_scatter(x=list(range(1, len(gap) + 1)), y=gap, mode="lines", name=m,
                            line=dict(color=MODEL_COLOR[m], width=2),
                            hovertemplate=f"<b>{m}</b><br>epoch %{{x}}: gap %{{y:.3f}}<extra></extra>")
        fig.add_hline(y=0, line_color=INK2, line_width=1)
        fig.update_layout(title="Generalisation gap (train acc − val acc)", xaxis_title="Epoch",
                          yaxis_title="Gap", yaxis_tickformat=".0%", hovermode="x unified")
        show(style_fig(fig, 380))
    with c2:
        fig = go.Figure()
        for m in selected:
            h = info[m]["history"]
            best_so_far = np.maximum.accumulate(h["val_accuracy"])
            fig.add_scatter(x=list(range(1, len(best_so_far) + 1)), y=best_so_far, mode="lines", name=m,
                            line=dict(color=MODEL_COLOR[m], width=2, shape="hv"),
                            hovertemplate=f"<b>{m}</b><br>epoch %{{x}}: %{{y:.3f}}<extra></extra>")
        fig.update_layout(title="Convergence speed (best val accuracy so far)", xaxis_title="Epoch",
                          yaxis_title="Val accuracy", yaxis_tickformat=".0%", hovermode="x unified")
        show(style_fig(fig, 380))

# =========================================================================== 3. per class
with tab_class:
    st.subheader("Per-class performance")
    metric_choice = st.radio("Metric", ["f1", "recall", "precision"], horizontal=True,
                             format_func=lambda s: {"f1": "F1", "recall": "Recall (class accuracy)",
                                                    "precision": "Precision"}[s])
    fig = go.Figure()
    for m in selected:
        fig.add_scatter(x=CLASSES, y=per_class[m][metric_choice], mode="lines+markers", name=m,
                        line=dict(color=MODEL_COLOR[m], width=2),
                        marker=dict(symbol=MODEL_SYMBOL[m], size=9, line=dict(color="white", width=1.5)),
                        hovertemplate=f"<b>{m}</b><br>%{{x}}: %{{y:.3f}}<extra></extra>")
    fig.update_layout(title=f"Per-class {metric_choice} profile", yaxis_title=metric_choice,
                      yaxis_range=[0, 1.02], hovermode="x unified")
    show(style_fig(fig, 400))

    c1, c2 = st.columns([1, 1])
    with c1:
        z = np.array([per_class[m][metric_choice] for m in selected])
        fig = go.Figure(go.Heatmap(z=z, x=CLASSES, y=selected, colorscale="Blues", zmin=0, zmax=1,
                                   text=np.round(z, 2), texttemplate="%{text}", xgap=2, ygap=2,
                                   hovertemplate="%{y} · %{x}: %{z:.3f}<extra></extra>",
                                   colorbar=dict(thickness=10)))
        fig.update_layout(title=f"{metric_choice.upper()} heat-map (model × class)", yaxis_autorange="reversed")
        show(style_fig(fig, 330, legend=False))
    with c2:
        cm_model = st.selectbox("Confusion matrix for", selected, key="cm_model")
        cm = confusion_matrix(y_true, probs[cm_model].argmax(1), labels=range(NC))
        cmn = cm / cm.sum(1, keepdims=True)
        fig = go.Figure(go.Heatmap(z=cmn, x=CLASSES, y=CLASSES, colorscale="Blues", zmin=0, zmax=1,
                                   customdata=cm, xgap=1, ygap=1, colorbar=dict(thickness=10),
                                   hovertemplate="true %{y} → pred %{x}<br>%{z:.1%} (%{customdata} imgs)<extra></extra>"))
        fig.update_layout(title=f"{cm_model}: row-normalised confusion matrix",
                          xaxis_title="Predicted", yaxis_title="True", yaxis_autorange="reversed")
        show(style_fig(fig, 330, legend=False))

    st.markdown("**Most-confused class pairs** (off-diagonal counts, all selected models)")
    pairs = []
    for m in selected:
        cm = confusion_matrix(y_true, probs[m].argmax(1), labels=range(NC))
        np.fill_diagonal(cm, 0)
        for i, j in zip(*np.unravel_index(np.argsort(cm, axis=None)[::-1][:5], cm.shape)):
            pairs.append(dict(Model=m, True_class=CLASSES[i], Predicted_as=CLASSES[j], Count=int(cm[i, j]),
                              Rate=100 * cm[i, j] / (y_true == i).sum()))
    st.dataframe(pd.DataFrame(pairs).rename(columns={"True_class": "True class", "Predicted_as": "Predicted as"}),
                 use_container_width=True, hide_index=True,
                 column_config={"Rate": st.column_config.NumberColumn("Rate (% of class)", format="%.1f%%")})

# =========================================================================== 4. calibration
with tab_cal:
    st.subheader("Calibration — can the confidence scores be trusted?")
    st.markdown('<div class="small-note">The main page reports a single confidence per prediction. '
                'A model is well calibrated when, among predictions made with 80% confidence, about 80% are '
                'correct. Points below the diagonal = over-confident.</div>', unsafe_allow_html=True)
    n_bins = st.slider("Number of bins", 5, 20, 10)
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                        line=dict(color=INK2, width=1, dash="dot"), hoverinfo="skip")
        for m in selected:
            e, rows = ece_score(probs[m], y_true, n_bins)
            cx, ay, n = zip(*rows)
            fig.add_scatter(x=cx, y=ay, mode="lines+markers", name=f"{m} (ECE {e:.3f})",
                            line=dict(color=MODEL_COLOR[m], width=2), customdata=n,
                            marker=dict(symbol=MODEL_SYMBOL[m], size=9, line=dict(color="white", width=1.5)),
                            hovertemplate=f"<b>{m}</b><br>mean conf %{{x:.2f}}<br>accuracy %{{y:.2f}}"
                                          f"<br>%{{customdata}} images<extra></extra>")
        fig.update_layout(title="Reliability diagram", xaxis_title="Mean predicted confidence",
                          yaxis_title="Observed accuracy", xaxis_range=[0, 1], yaxis_range=[0, 1.02])
        show(style_fig(fig, 430))
    with c2:
        conf_model = st.selectbox("Confidence distribution for", selected, key="conf_model")
        conf = probs[conf_model].max(1)
        ok = probs[conf_model].argmax(1) == y_true
        fig = go.Figure()
        for flag, color, lbl in ((True, MODEL_COLOR[conf_model], "Correct"), (False, INK2, "Wrong")):
            fig.add_histogram(x=conf[ok == flag], name=f"{lbl} ({(ok == flag).sum():,})", xbins=dict(start=0, end=1, size=0.05),
                              histnorm="probability", marker_color=color, opacity=0.75,
                              marker_line=dict(color="white", width=1),
                              hovertemplate=f"{lbl}<br>conf %{{x}}<br>share %{{y:.1%}}<extra></extra>")
        fig.update_layout(barmode="overlay", title=f"{conf_model}: confidence of correct vs wrong predictions",
                          xaxis_title="Max softmax confidence", yaxis_title="Share of predictions",
                          yaxis_tickformat=".0%", xaxis_range=[0, 1])
        show(style_fig(fig, 430))
    cal = metrics.loc[selected, ["ECE", "NLL", "Brier"]].copy()
    cal["Mean confidence"] = [probs[m].max(1).mean() for m in selected]
    cal["Accuracy"] = metrics.loc[selected, "Accuracy"]
    cal["Over-confidence"] = cal["Mean confidence"] - cal["Accuracy"]
    st.dataframe(cal.style.format("{:.3f}"), use_container_width=True)

# =========================================================================== 5. ROC
with tab_roc:
    st.subheader("ROC curves (one-vs-rest)")
    roc_class = st.selectbox("Class", ["Macro-average"] + CLASSES)
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(color=INK2, width=1, dash="dot"),
                    hoverinfo="skip")
    grid = np.linspace(0, 1, 201)
    for m in selected:
        if roc_class == "Macro-average":
            tprs = []
            for c in range(NC):
                fpr, tpr, _ = roc_curve(y_true == c, probs[m][:, c])
                tprs.append(np.interp(grid, fpr, tpr))
            fx, ty = grid, np.mean(tprs, 0)
            ty[0] = 0
        else:
            c = CLASSES.index(roc_class)
            fx, ty, _ = roc_curve(y_true == c, probs[m][:, c])
        fig.add_scatter(x=fx, y=ty, mode="lines", name=f"{m} (AUC {auc(fx, ty):.3f})",
                        line=dict(color=MODEL_COLOR[m], width=2),
                        hovertemplate=f"<b>{m}</b><br>FPR %{{x:.3f}}<br>TPR %{{y:.3f}}<extra></extra>")
    fig.update_layout(title=f"ROC — {roc_class}", xaxis_title="False positive rate",
                      yaxis_title="True positive rate", xaxis_range=[0, 1], yaxis_range=[0, 1.02])
    show(style_fig(fig, 480))

# =========================================================================== 6. robustness
with tab_rob:
    st.subheader("Robustness to input noise")
    st.markdown(f'<div class="small-note">Accuracy on {meta["n_robust"]:,} test images with additive Gaussian '
                'noise (σ on the 0–1 pixel scale). A flatter curve means the model degrades more gracefully on '
                'real-world, noisy or uploaded images.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for m in selected:
            r = info[m]["robustness"]
            xs = sorted(float(s) for s in r)
            ys = [r[next(k_ for k_ in r if float(k_) == s)] for s in xs]
            fig.add_scatter(x=xs, y=ys, mode="lines+markers", name=m, line=dict(color=MODEL_COLOR[m], width=2),
                            marker=dict(symbol=MODEL_SYMBOL[m], size=9, line=dict(color="white", width=1.5)),
                            hovertemplate=f"<b>{m}</b><br>σ=%{{x}}: %{{y:.1%}}<extra></extra>")
        fig.update_layout(title="Accuracy vs noise level", xaxis_title="Gaussian noise σ",
                          yaxis_title="Accuracy", yaxis_tickformat=".0%", hovermode="x unified")
        show(style_fig(fig, 420))
    with c2:
        fig = go.Figure()
        for m in selected:
            r = info[m]["robustness"]
            xs = sorted(float(s) for s in r)
            ys = np.array([r[next(k_ for k_ in r if float(k_) == s)] for s in xs])
            fig.add_scatter(x=xs, y=ys / ys[0], mode="lines+markers", name=m,
                            line=dict(color=MODEL_COLOR[m], width=2),
                            marker=dict(symbol=MODEL_SYMBOL[m], size=9, line=dict(color="white", width=1.5)),
                            hovertemplate=f"<b>{m}</b><br>σ=%{{x}}: %{{y:.1%}} of clean<extra></extra>")
        fig.update_layout(title="Relative accuracy retained (clean = 100%)", xaxis_title="Gaussian noise σ",
                          yaxis_title="Share of clean accuracy", yaxis_tickformat=".0%", hovermode="x unified")
        show(style_fig(fig, 420))

# =========================================================================== 7. efficiency
with tab_eff:
    st.subheader("Accuracy vs computational cost")
    c1, c2 = st.columns(2)
    xs = {m: info[m]["latency_ms_bs1"] for m in selected}
    pareto = [m for m in selected
              if not any(xs[o] <= xs[m] and metrics.loc[o, "Accuracy"] >= metrics.loc[m, "Accuracy"]
                         and o != m and (xs[o] < xs[m] or metrics.loc[o, "Accuracy"] > metrics.loc[m, "Accuracy"])
                         for o in selected)]
    with c1:
        fig = go.Figure()
        pf = sorted(pareto, key=lambda m: xs[m])
        fig.add_scatter(x=[xs[m] for m in pf], y=[metrics.loc[m, "Accuracy"] for m in pf], mode="lines",
                        name="Pareto frontier", line=dict(color=INK2, width=1.5, dash="dot"), hoverinfo="skip")
        for m in selected:
            p_m = info[m]["total_params"] / 1e6
            fig.add_scatter(x=[xs[m]], y=[metrics.loc[m, "Accuracy"]], mode="markers+text", name=m,
                            text=[m], textposition="top center", textfont=dict(color=INK),
                            marker=dict(color=MODEL_COLOR[m], symbol=MODEL_SYMBOL[m],
                                        size=14 + 3 * np.sqrt(p_m), line=dict(color="white", width=2)),
                            hovertemplate=f"<b>{m}</b><br>latency %{{x:.1f}} ms<br>acc %{{y:.3f}}"
                                          f"<br>{p_m:.1f} M params<extra></extra>")
        fig.update_layout(title="Accuracy vs latency (marker size ∝ √params)",
                          xaxis_title="Latency per image, batch = 1 (ms, CPU)", yaxis_title="Test accuracy",
                          xaxis_type="log")
        show(style_fig(fig, 430))
    with c2:
        eff_df = pd.DataFrame({
            "Latency bs=1 (ms)": [info[m]["latency_ms_bs1"] for m in selected],
            "Throughput bs=64 (img/s)": [1000 / info[m]["latency_ms_per_img_bs64"] for m in selected],
            "Total params (M)": [info[m]["total_params"] / 1e6 for m in selected],
            "Approx. size fp32 (MB)": [info[m]["total_params"] * 4 / 1e6 for m in selected],
            "Training time (s)": [info[m]["fit_time_s"] + info[m].get("feature_time_s", 0) for m in selected],
            "Acc. per 10 ms": [metrics.loc[m, "Accuracy"] / info[m]["latency_ms_bs1"] * 10 for m in selected],
            "On Pareto frontier": ["✅" if m in pareto else "—" for m in selected],
        }, index=pd.Index(selected, name="Model"))
        st.dataframe(eff_df.style.format({c: "{:.2f}" for c in eff_df.columns if c != "On Pareto frontier"}),
                     use_container_width=True)
        st.caption(f"Latency measured on: {meta['hardware']}. Streamlit Cloud hardware will differ; the ranking "
                   "is what matters.")

# =========================================================================== 8. significance & ensemble
with tab_stats:
    st.subheader("Are the differences statistically significant?")
    st.markdown('<div class="small-note">Exact McNemar test on paired predictions (same test images), '
                'Holm-corrected for multiple comparisons. Bootstrap CIs alone can overlap even when a paired '
                'test is significant, so both are shown.</div>', unsafe_allow_html=True)
    correct = {m: probs[m].argmax(1) == y_true for m in selected}
    prs = list(combinations(selected, 2))
    if prs:
        raw, rows = [], []
        for a, b in prs:
            b01 = int(np.sum(correct[a] & ~correct[b]))
            b10 = int(np.sum(~correct[a] & correct[b]))
            p = binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 > 0 else 1.0
            raw.append(p)
            rows.append(dict(A=a, B=b, **{"A right, B wrong": b01, "A wrong, B right": b10,
                                          "Δ accuracy (A−B)": correct[a].mean() - correct[b].mean()}))
        adj = holm(np.array(raw))
        for r, p, pa in zip(rows, raw, adj):
            r["p (raw)"], r["p (Holm)"] = p, pa
            r["Significant (α=0.05)"] = "✅" if pa < 0.05 else "—"
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, column_config={
            "Δ accuracy (A−B)": st.column_config.NumberColumn(format="%.3f"),
            "p (raw)": st.column_config.NumberColumn(format="%.2e"),
            "p (Holm)": st.column_config.NumberColumn(format="%.2e")})
        c2, c3 = st.columns([1, 1])
        with c3:
            st.markdown("**How to read this**")
            st.markdown("- *A right, B wrong* / *A wrong, B right* are the discordant pairs McNemar uses.\n"
                        "- Only images where the two models disagree carry information about which is better.\n"
                        "- A difference can be statistically significant yet too small to matter in practice — "
                        "check Δ accuracy alongside p.")
        with c2:
            n = len(selected)
            mat = np.full((n, n), np.nan)
            for (a, b), pa in zip(prs, adj):
                i, j = selected.index(a), selected.index(b)
                mat[i, j] = mat[j, i] = -np.log10(max(pa, 1e-300))
            fig = go.Figure(go.Heatmap(z=mat, x=selected, y=selected, colorscale="Blues", xgap=2, ygap=2,
                                       text=np.where(np.isnan(mat), "", np.round(mat, 1)), texttemplate="%{text}",
                                       hovertemplate="%{y} vs %{x}<br>−log10 p = %{z:.2f}<extra></extra>",
                                       colorbar=dict(title="−log₁₀ p", thickness=10)))
            fig.update_layout(title="Pairwise significance (Holm, −log₁₀ p; >1.3 ⇒ p<0.05)",
                              yaxis_autorange="reversed")
            show(style_fig(fig, 340, legend=False))

    st.subheader("Do the models make the same mistakes?")
    c1, c2 = st.columns(2)
    with c1:
        n = len(selected)
        jac = np.eye(n)
        for i, a in enumerate(selected):
            for j, b in enumerate(selected):
                if i != j:
                    ea, eb = ~correct[a], ~correct[b]
                    jac[i, j] = (ea & eb).sum() / max((ea | eb).sum(), 1)
        fig = go.Figure(go.Heatmap(z=jac, x=selected, y=selected, colorscale="Blues", zmin=0, zmax=1,
                                   text=np.round(jac, 2), texttemplate="%{text}", xgap=2, ygap=2,
                                   hovertemplate="%{y} & %{x}<br>error overlap (Jaccard) %{z:.2f}<extra></extra>",
                                   colorbar=dict(thickness=10)))
        fig.update_layout(title="Error overlap (Jaccard of misclassified sets)", yaxis_autorange="reversed")
        show(style_fig(fig, 360, legend=False))
    with c2:
        ens_prob = np.mean([probs[m] for m in selected], 0)
        ens_acc = (ens_prob.argmax(1) == y_true).mean()
        oracle = np.any([correct[m] for m in selected], 0).mean()
        all_wrong = np.all([~correct[m] for m in selected], 0).mean()
        best_single = metrics.loc[selected, "Accuracy"].max()
        lbls = ["Best single model", "Soft-vote ensemble", "Oracle (any model right)"]
        vals = [best_single, ens_acc, oracle]
        fig = go.Figure(go.Bar(x=vals, y=lbls, orientation="h", marker_color=["#2a78d6", "#6250d6", "#52514e"],
                               marker_line=dict(color="white", width=2), text=[f"{v:.1%}" for v in vals],
                               textposition="outside", hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig.update_layout(title="Complementarity: ensemble headroom", xaxis_range=[0, 1.08],
                          xaxis_tickformat=".0%", yaxis_autorange="reversed")
        show(style_fig(fig, 360, legend=False))
        st.caption(f"{all_wrong:.1%} of test images are misclassified by every selected model — "
                   "likely ambiguous or mislabelled images worth inspecting.")

# =========================================================================== 9. deployment audit
with tab_audit:
    st.subheader("Gap analysis: benchmark vs. what the main page actually serves")
    if audit:
        rows = audit.get("served_models", [])
        if rows:
            adf = pd.DataFrame(rows).set_index("model")
            adf = adf.rename(columns={"accuracy": "Accuracy as served", "weights": "Weights", "params": "Params",
                                      "app_label_params": "Params shown in app", "preprocessing": "Preprocessing",
                                      "architecture_matches_app_code": "Matches app code",
                                      "distinct_classes_predicted": "Distinct classes predicted",
                                      "mean_confidence": "Mean confidence", "n_images": "Test images"})
            first = [c for c in ["Accuracy as served", "Weights", "Distinct classes predicted", "Params",
                                 "Params shown in app", "Preprocessing"] if c in adf.columns]
            adf = adf[first + [c for c in adf.columns if c not in first]]
            bench_acc = {m: metrics.loc[m, "Accuracy"] for m in models}
            adf.insert(1, "Benchmark accuracy (trained)", [bench_acc.get(m, np.nan) for m in adf.index])
            st.dataframe(adf, use_container_width=True, column_config={
                "Accuracy as served": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1),
                "Benchmark accuracy (trained)": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1),
                "Params": st.column_config.NumberColumn(format="%d"),
                "Mean confidence": st.column_config.NumberColumn(format="%.3f")})
        for f_ in audit.get("findings", []):
            st.markdown(f'<div class="finding {"warn" if f_.get("severity") == "high" else ""}">'
                        f'<b>{f_["title"]}</b><br>{f_["detail"]}</div>', unsafe_allow_html=True)
        st.caption(f"Audit generated {audit.get('created_utc', '')} on {audit.get('n_images', '?')} test images.")
    else:
        st.info("Run `python benchmark/audit_deployed.py` to evaluate the models exactly as app.py serves them.")

    st.markdown("**Benchmark caveats**")
    st.markdown(f"""
- Subset protocol ({meta['n_train']:,} training images) so all four models train on identical data on a CPU; full-data
  training will raise every model's accuracy, and the CNN typically gains the most.
- Transfer models use frozen ImageNet backbones at {meta['transfer_input_size']}px with their own `preprocess_input`;
  fine-tuning the top blocks or using 224px would raise their accuracy further.
- No data augmentation is used, to keep the comparison like-for-like.
- Data source: `{meta['data_source']}`.
""")

st.markdown("---")
st.markdown("<div style='text-align:center;color:#666;font-size:.85rem'>Model Comparison · generated from "
            "<code>results/</code> by <code>benchmark/train_and_benchmark.py</code></div>",
            unsafe_allow_html=True)
