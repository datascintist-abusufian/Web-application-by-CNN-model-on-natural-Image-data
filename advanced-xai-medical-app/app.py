"""
CIFAR-10 Research Classifier — main Streamlit page.

Four models (compact CNN, and frozen ResNet50 / VGG16 / MobileNetV2 backbones with heads trained by
benchmark/train_and_benchmark.py). For every prediction the page shows confidence and entropy,
six attribution methods with deletion/insertion faithfulness, test-time-augmentation and
MC-Dropout uncertainty, a corruption / adversarial probe, session metrics on labelled images
only, and a batch evaluation on a 100-image labelled gallery from the CIFAR-10 test set.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from plotly.subplots import make_subplots
from scipy.stats import entropy, spearmanr
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Advanced CIFAR-10 Research Classifier", page_icon="🧠", layout="wide",
                   initial_sidebar_state="expanded")

ROOT = Path(__file__).resolve().parent
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
MODELS = ["CNN", "ResNet50", "VGG16", "MobileNetV2"]
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a5cd6", "#d63a7a", "#3bb3c4", "#7a7a72", "#a0522d", "#4b5563"]
METHOD_COLOR = {"Grad-CAM": "#eb6834", "Grad-CAM++": "#d63a7a", "Saliency": "#2a78d6", "SmoothGrad": "#3bb3c4",
                "Integrated Gradients": "#1baf7a", "Occlusion": "#eda100"}
INK2, GRID = "#52514e", "#e6e5e0"

st.markdown("""
<style>
.gradient-header{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:1.6rem 2rem;
  border-radius:15px;margin:.5rem 0 1rem;text-align:center}
.gradient-header h1{color:#fff;margin:0;font-size:2rem}
.gradient-header p{margin:.35rem 0 0;opacity:.92}
.prediction-high{padding:1rem;background:#d4edda;border-radius:10px;border-left:5px solid #28a745;margin:.5rem 0}
.prediction-low{padding:1rem;background:#f8d7da;border-radius:10px;border-left:5px solid #dc3545;margin:.5rem 0}
.prediction-neutral{padding:1rem;background:#e7f1ff;border-radius:10px;border-left:5px solid #2a78d6;margin:.5rem 0}
.stButton>button{width:100%;background:linear-gradient(135deg,#667eea,#764ba2);color:white;border-radius:10px;
  padding:.6rem 1.2rem;font-weight:bold;border:none}
.info-box{padding:1rem;background:#d1ecf1;border-radius:10px;border-left:5px solid #17a2b8;margin:.6rem 0;font-size:.9rem}
.finding{background:#f8f9fa;border-radius:10px;padding:.7rem 1rem;margin:.35rem 0;border-left:4px solid #17a2b8}
.finding.warn{border-left-color:#dc3545;background:#fff5f5}
</style>""", unsafe_allow_html=True)


# =========================================================================== helpers
def style_fig(fig, height=400, legend=True):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=50, b=10),
                      showlegend=legend, legend=dict(orientation="h", yanchor="top", y=-0.2, x=0, title=None))
    if fig.layout.title.text:
        fig.update_layout(title=dict(x=0, xanchor="left", font=dict(size=15)))
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def show(fig):
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def finding(text, kind=""):
    st.markdown(f'<div class="finding {kind}">{text}</div>', unsafe_allow_html=True)


def u8(x01):
    return (np.clip(x01, 0, 1) * 255).round().astype(np.uint8)


def big(x01, size=224, smooth=False):
    return Image.fromarray(u8(x01)).resize((size, size), Image.BILINEAR if smooth else Image.NEAREST)


@st.cache_data(show_spinner=False)
def load_gallery():
    d = np.load(ROOT / "sample_data" / "cifar10_test_gallery.npz")
    return d["images"].astype(np.float32) / 255.0, d["labels"].astype(int), d["test_index"]


@st.cache_data(show_spinner=False)
def load_class_sample(cls):
    p = ROOT / f"cifar_image_{cls}_1.png"
    return np.asarray(Image.open(p).convert("RGB").resize((32, 32))).astype(np.float32) / 255.0


@st.cache_resource(show_spinner="Loading model (first load downloads ImageNet weights)…")
def get_model(name):
    import cifar_models
    return cifar_models.load(name)


@st.cache_data(show_spinner=False)
def served_accuracy():
    import json
    p = ROOT / "results" / "deployment_audit.json"
    if not p.exists():
        return {}
    a = json.load(open(p))
    return {m["model"]: m for m in a.get("served_models", [])}


def init_state():
    for k, v in dict(model_type=None, history=[], last=None).items():
        st.session_state.setdefault(k, v)


init_state()

# =========================================================================== header + sidebar
st.markdown("""<div class="gradient-header"><h1>🧠 Advanced CIFAR-10 Research Classifier</h1>
<p>Multi-model prediction · explainability · uncertainty · robustness probes</p></div>""", unsafe_allow_html=True)

with st.sidebar:
    st.header("🎛️ Research controls")
    model_type = st.selectbox("Model architecture", MODELS,
                              index=MODELS.index(st.session_state.model_type) if st.session_state.model_type else 0)
    if st.button("🔄 Load model", use_container_width=True) or (
            st.session_state.model_type is not None and st.session_state.model_type != model_type):
        try:
            get_model(model_type)
            st.session_state.model_type = model_type
            st.session_state.last = None
        except FileNotFoundError as e:
            st.error(f"Weights missing: {e}. Run benchmark/train_and_benchmark.py --save-weights.")
    if st.session_state.model_type:
        ex, info = get_model(st.session_state.model_type)
        audit = served_accuracy().get(st.session_state.model_type, {})
        acc_txt = f"<br><strong>Served test accuracy:</strong> {audit['accuracy']:.1%} (n={audit['n_images']:,})" if audit else ""
        st.markdown(f"""<div class="info-box"><strong>📋 {st.session_state.model_type}</strong> ✅ loaded<br>
            <strong>Parameters:</strong> {info['params']:,}<br><strong>Input:</strong> {info['input']}<br>
            <strong>Weights:</strong> {info['weights']}<br>{info['desc']}{acc_txt}</div>""", unsafe_allow_html=True)
        labelled = [h for h in st.session_state.history if h["label"] is not None]
        if labelled:
            st.metric("Session accuracy (labelled only)",
                      f"{np.mean([h['label'] == h['pred'] for h in labelled]):.1%}", f"{len(labelled)} images")
        if st.button("🗑️ Reset session history", use_container_width=True):
            st.session_state.history = []
    else:
        st.warning("No model loaded yet.")
    st.markdown("---")
    st.caption("Other pages: **Model Comparison** (4-model benchmark) and **Medical Imaging** "
               "(chest X-ray & blood-cell analysis).")

# =========================================================================== no model → gallery
gal_x, gal_y, gal_idx = load_gallery()
if not st.session_state.model_type:
    st.info("👈 Choose a model and click **Load model**. Below: all ten CIFAR-10 classes.")
    cols = st.columns(10)
    for c, col in enumerate(cols):
        col.image(big(load_class_sample(CLASS_NAMES[c]), 96), caption=CLASS_NAMES[c].capitalize())
    st.markdown("##### Labelled test gallery (10 images per class, official CIFAR-10 test set)")
    for c in range(10):
        cols = st.columns(11)
        cols[0].markdown(f"**{CLASS_NAMES[c]}**")
        for j, i in enumerate(np.where(gal_y == c)[0][:10]):
            cols[j + 1].image(big(gal_x[i], 64))
    st.stop()

ex, info = get_model(st.session_state.model_type)
import xai_utils as X  # noqa: E402  (TensorFlow is already imported by the model)

# =========================================================================== image selection
st.subheader("🔍 Image selection")
c1, c2, c3 = st.columns([1.2, 1.2, 1])
with c1:
    source = st.radio("Source", ["Class sample", "Test gallery (100 labelled)", "Upload your own (unlabelled)"])
with c2:
    label, img = None, None
    if source == "Class sample":
        cls = st.selectbox("Class", CLASS_NAMES, format_func=str.capitalize)
        img, label = load_class_sample(cls), CLASS_NAMES.index(cls)
    elif source.startswith("Test gallery"):
        cls = st.selectbox("Class", CLASS_NAMES, format_func=str.capitalize)
        ids = np.where(gal_y == CLASS_NAMES.index(cls))[0]
        k = st.slider("Image #", 1, len(ids), 1)
        img, label = gal_x[ids[k - 1]], int(gal_y[ids[k - 1]])
        st.caption(f"CIFAR-10 test index {gal_idx[ids[k - 1]]}")
    else:
        up = st.file_uploader("Image", type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"])
        if up is not None:
            import cifar_models
            img = cifar_models.to_cifar_input(Image.open(up))
            known = st.selectbox("True label (optional — only used for session metrics)",
                                 ["unknown"] + CLASS_NAMES)
            label = None if known == "unknown" else CLASS_NAMES.index(known)
with c3:
    if img is not None:
        st.image(big(img, 200), caption="Model input (32×32)")

run = st.button("🔬 Analyze image", type="primary", use_container_width=True, disabled=img is None)
if run:
    t0 = time.perf_counter()
    probs = ex.predict(img[None])[0]
    dt = (time.perf_counter() - t0) * 1000
    st.session_state.last = dict(img=img, label=label, probs=probs, ms=dt, model=st.session_state.model_type)
    st.session_state.history.append(dict(label=label, pred=int(probs.argmax()), conf=float(probs.max()), ms=dt,
                                         model=st.session_state.model_type))

last = st.session_state.last
if last is None or last["model"] != st.session_state.model_type:
    st.info("Select an image and click **Analyze image**.")
    st.stop()

img, label, probs = last["img"], last["label"], last["probs"]
pred = int(probs.argmax())
H = entropy(probs, base=2)

# =========================================================================== results
c1, c2, c3, c4 = st.columns(4)
with c1:
    if label is None:
        st.markdown(f'<div class="prediction-neutral"><strong>Prediction:</strong> {CLASS_NAMES[pred]}<br>'
                    f'<small>unlabelled image — not counted in accuracy</small></div>', unsafe_allow_html=True)
    elif label == pred:
        st.markdown(f'<div class="prediction-high"><strong>✅ Correct</strong><br>Predicted: {CLASS_NAMES[pred]}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="prediction-low"><strong>❌ Incorrect</strong><br>True: {CLASS_NAMES[label]}<br>'
                    f'Predicted: {CLASS_NAMES[pred]}</div>', unsafe_allow_html=True)
c2.metric("Confidence", f"{probs.max():.1%}")
c3.metric("Entropy", f"{H:.3f} bits", help=f"{H / np.log2(10):.0%} of the maximum (uniform over 10 classes = 3.32 bits)")
c3.caption(f"{H / np.log2(10):.0%} of maximum entropy")
c4.metric("Inference time", f"{last['ms']:.1f} ms")

tabs = st.tabs(["📊 Prediction", "🔬 Explainability", "🎲 Uncertainty", "🛡️ Robustness probe",
                "📈 Session metrics", "🧪 Batch evaluation"])

# ---------------------------------------------------------------- prediction
with tabs[0]:
    fig = make_subplots(rows=1, cols=2, column_widths=[.65, .35],
                        subplot_titles=["Confidence by class", "Sorted confidence (rank)"])
    fig.add_bar(x=CLASS_NAMES, y=probs * 100, marker_color=[PALETTE[1] if i == pred else PALETTE[0] for i in range(10)],
                text=[f"{v * 100:.1f}%" for v in probs], textposition="outside", row=1, col=1)
    fig.add_scatter(x=np.arange(1, 11), y=np.sort(probs)[::-1] * 100, mode="lines+markers",
                    line=dict(color="#667eea"), row=1, col=2)
    fig.update_yaxes(title_text="%", row=1, col=1)
    show(style_fig(fig, 380, legend=False))
    top_k = st.slider("Top-k predictions", 1, 10, 5)
    top = np.argsort(probs)[::-1][:top_k]
    st.dataframe(pd.DataFrame({"Rank": range(1, top_k + 1), "Class": [CLASS_NAMES[i].capitalize() for i in top],
                               "Confidence": [f"{probs[i] * 100:.2f}%" for i in top],
                               "Cumulative": [f"{probs[top[:r + 1]].sum() * 100:.1f}%" for r in range(top_k)]}),
                 hide_index=True, use_container_width=True)
    margin = np.sort(probs)[-1] - np.sort(probs)[-2]
    finding(f"Top-1 vs top-2 margin: <b>{margin:.1%}</b> ({CLASS_NAMES[np.argsort(probs)[-2]]} is the runner-up). "
            "Small margins and high entropy mark ambiguous inputs.", "warn" if margin < .2 else "")

# ---------------------------------------------------------------- explainability
with tabs[1]:
    @st.cache_data(show_spinner="Computing six attribution maps…")
    def explain(model_name, img_bytes, cls):
        x = np.frombuffer(img_bytes, np.float32).reshape(32, 32, 3)
        e, _ = get_model(model_name)
        heavy = model_name != "CNN"  # ImageNet backbones at 96 px: use lighter settings on CPU
        maps, _ = X.all_maps(e, x, cls, fast=heavy)
        di = {m: X.deletion_insertion(e, x, mp, cls, steps=10 if heavy else 20) for m, mp in maps.items()}
        return maps, di

    target = st.selectbox("Explain class", CLASS_NAMES, index=pred, format_func=lambda c: f"{c} (p = {probs[CLASS_NAMES.index(c)]:.1%})")
    tc = CLASS_NAMES.index(target)
    maps, di = explain(last["model"], img.astype(np.float32).tobytes(), tc)
    cols = st.columns(len(maps) + 1)
    cols[0].image(big(img, 150), caption="Input")
    for col, (m, mp) in zip(cols[1:], maps.items()):
        col.image(big(X.overlay(img, mp), 150, smooth=True), caption=m)
    st.caption("Grad-CAM uses the last convolutional block (CNN: 13×13; ResNet50/MobileNetV2: 3×3; VGG16: 6×6 at 96 px), "
               "so it is coarse; gradient-based maps are pixel-level. Occlusion slides a 4×4 grey patch.")
    c1, c2 = st.columns(2)
    with c1:
        dfm = pd.DataFrame([{"Method": m, "Deletion AUC ↓": round(v[3], 3), "Insertion AUC ↑": round(v[4], 3),
                             "Ins − Del ↑": round(v[4] - v[3], 3)} for m, v in di.items()]).sort_values("Ins − Del ↑", ascending=False)
        st.dataframe(dfm, hide_index=True, use_container_width=True)
        fig = go.Figure()
        for m, v in di.items():
            fig.add_scatter(x=v[0], y=v[1], name=m, line=dict(color=METHOD_COLOR[m]))
        fig.update_layout(title="Deletion curves (faster drop = more faithful)", xaxis_title="Fraction of pixels removed",
                          yaxis_title=f"P({target})")
        show(style_fig(fig, 360))
    with c2:
        names = list(maps)
        R = np.array([[spearmanr(maps[a].ravel(), maps[b].ravel())[0] for b in names] for a in names])
        fig = px.imshow(R, x=names, y=names, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu",
                        title="Agreement between methods (Spearman ρ)")
        show(style_fig(fig, 470, legend=False))
    finding(f"Most faithful method for this image: <b>{dfm.iloc[0]['Method']}</b>. Low agreement between methods is "
            "common and is a reason to report more than one explanation (Krishna et al., 2022, “The disagreement problem”).")

# ---------------------------------------------------------------- uncertainty
with tabs[2]:
    @st.cache_data(show_spinner="Sampling…")
    def uncertainty(model_name, img_bytes):
        x = np.frombuffer(img_bytes, np.float32).reshape(1, 32, 32, 3)
        e, inf = get_model(model_name)
        tta = X.tta_samples(e, x, n=24, flips=True, max_shift=0.1, noise=0.02)[:, 0]
        mc = X.mc_dropout(e, x, T=40)
        return tta, (mc[:, 0] if mc is not None else None)

    tta, mc = uncertainty(last["model"], img.astype(np.float32).tobytes())
    rows = []
    for nm, s in [("Test-time augmentation (24 views)", tta), ("MC-Dropout (40 passes)", mc)]:
        if s is None:
            continue
        pm, tot, ale, epi = X.decompose(s[:, None, :])
        rows.append({"Method": nm, "Mean prediction": CLASS_NAMES[pm[0].argmax()], "Mean confidence": f"{pm[0].max():.1%}",
                     "Total (nats)": round(float(tot[0]), 4), "Aleatoric": round(float(ale[0]), 4),
                     "Epistemic (MI)": round(float(epi[0]), 4),
                     "Label flip rate": f"{(s.argmax(1) != pred).mean():.0%}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    if mc is None:
        st.caption("The compact CNN has no dropout layer, so MC-Dropout is not available for it; "
                   "TTA captures input-perturbation uncertainty instead.")
    fig = go.Figure()
    for nm, s, col in [("TTA", tta, PALETTE[0]), ("MC-Dropout", mc, PALETTE[1])]:
        if s is None:
            continue
        fig.add_box(x=[CLASS_NAMES[c] for c in range(10) for _ in range(len(s))], y=np.concatenate([s[:, c] for c in range(10)]),
                    name=nm, marker_color=col, boxpoints=False)
    fig.update_layout(boxmode="group", title="Predictive distribution per class", yaxis_title="Probability")
    show(style_fig(fig, 400))
    finding("Aleatoric = expected entropy of each sample (ambiguity in the image); epistemic = mutual information "
            "between the prediction and the sampled model/view (what more data would reduce).")

# ---------------------------------------------------------------- robustness probe
with tabs[3]:
    c1, c2 = st.columns(2)
    with c1:
        kind = st.selectbox("Corruption", X.CORRUPTIONS)
        sev = st.slider("Severity", 1, 5, 3)
        xc = X.corrupt(img[None], kind, sev).astype(np.float32)
        pc = ex.predict(xc)[0]
        st.image([big(img, 160), big(xc[0], 160)], caption=["clean", f"{kind} s{sev}"])
        st.markdown(f"Prediction: **{CLASS_NAMES[pc.argmax()]}** ({pc.max():.1%}) — "
                    f"{'unchanged ✅' if pc.argmax() == pred else 'changed ⚠️'}")
        sweep = {k_: [ex.predict(X.corrupt(img[None], k_, s_).astype(np.float32))[0][pred] for s_ in range(1, 6)]
                 for k_ in X.CORRUPTIONS}
        fig = go.Figure()
        for i, (k_, v) in enumerate(sweep.items()):
            fig.add_scatter(x=list(range(0, 6)), y=[probs[pred]] + v, name=k_, line=dict(color=PALETTE[i]))
        fig.update_layout(title=f"P({CLASS_NAMES[pred]}) vs severity", xaxis_title="severity", yaxis_range=[0, 1])
        show(style_fig(fig, 340))
    with c2:
        eps = st.slider("FGSM ε (grey levels out of 255)", 0.0, 16.0, 2.0, 0.5)
        yt = np.array([label if label is not None else pred])
        xa = X.fgsm(ex, img[None], yt, eps / 255.0) if eps > 0 else img[None]
        pa = ex.predict(xa.astype(np.float32))[0]
        pert = 0.5 + (xa[0] - img) / (2 * max(eps, 1e-6) / 255.0)
        st.image([big(img, 160), big(xa[0], 160), big(pert, 160)],
                 caption=["clean", f"adversarial ε={eps}", "signed perturbation (grey = 0)"])
        st.markdown(f"Prediction: **{CLASS_NAMES[pa.argmax()]}** ({pa.max():.1%}) — "
                    f"{'unchanged ✅' if pa.argmax() == pred else 'flipped ⚠️'}")
        es = [0, 0.5, 1, 2, 4, 8, 16]
        curve = [ex.predict(X.fgsm(ex, img[None], yt, e / 255).astype(np.float32))[0][yt[0]] if e else probs[yt[0]] for e in es]
        fig = go.Figure(go.Scatter(x=es, y=curve, mode="lines+markers", line=dict(color=PALETTE[1], width=2.5)))
        fig.update_layout(title=f"P({CLASS_NAMES[yt[0]]}) under FGSM", xaxis_title="ε / 255", yaxis_range=[0, 1])
        show(style_fig(fig, 340, legend=False))

# ---------------------------------------------------------------- session metrics
with tabs[4]:
    hist = pd.DataFrame(st.session_state.history)
    lab = hist[hist["label"].notna()] if len(hist) else hist
    c = st.columns(4)
    c[0].metric("Predictions", len(hist))
    c[1].metric("Labelled", len(lab))
    c[2].metric("Accuracy (labelled)", f"{(lab['label'] == lab['pred']).mean():.1%}" if len(lab) else "—")
    c[3].metric("Mean inference", f"{hist['ms'].mean():.1f} ms" if len(hist) else "—")
    if len(lab) >= 2:
        cm = confusion_matrix(lab["label"].astype(int), lab["pred"].astype(int), labels=range(10))
        fig = px.imshow(cm, x=CLASS_NAMES, y=CLASS_NAMES, text_auto=True, color_continuous_scale="Blues",
                        title="Session confusion matrix (labelled images only; rows = truth)")
        show(style_fig(fig, 460, legend=False))
    if len(hist):
        hist_disp = hist.assign(label=hist["label"].map(lambda v: "unlabelled" if pd.isna(v) else CLASS_NAMES[int(v)]),
                                pred=hist["pred"].map(lambda v: CLASS_NAMES[int(v)]))
        st.dataframe(hist_disp, use_container_width=True)

# ---------------------------------------------------------------- batch evaluation
with tabs[5]:
    st.markdown(f"Runs **{st.session_state.model_type}** on all 100 labelled gallery images (10 per class, "
                "official CIFAR-10 test set) and reports accuracy, calibration and per-class recall.")
    if st.button("▶️ Run batch evaluation"):
        P = ex.predict(gal_x)
        st.session_state[f"batch_{st.session_state.model_type}"] = P
    P = st.session_state.get(f"batch_{st.session_state.model_type}")
    if P is not None:
        cal = X.calibration_stats(P, gal_y, n_bins=10)
        pr = P.argmax(1)
        c = st.columns(4)
        c[0].metric("Accuracy", f"{(pr == gal_y).mean():.1%}")
        c[1].metric("Top-3 accuracy", f"{np.mean([gal_y[i] in np.argsort(P[i])[-3:] for i in range(len(P))]):.1%}")
        c[2].metric("ECE (10 bins)", f"{cal['ece']:.3f}")
        c[3].metric("NLL", f"{cal['nll']:.3f}")
        c1, c2 = st.columns(2)
        with c1:
            rec = [(pr[gal_y == k_] == k_).mean() for k_ in range(10)]
            fig = go.Figure(go.Bar(x=CLASS_NAMES, y=rec, marker_color=PALETTE[0], text=[f"{r:.0%}" for r in rec],
                                   textposition="outside"))
            fig.update_layout(title="Per-class recall", yaxis_range=[0, 1.1])
            show(style_fig(fig, 380, legend=False))
        with c2:
            cm = confusion_matrix(gal_y, pr, labels=range(10))
            fig = px.imshow(cm, x=CLASS_NAMES, y=CLASS_NAMES, text_auto=True, color_continuous_scale="Blues",
                            title="Confusion matrix (rows = truth)")
            show(style_fig(fig, 380, legend=False))
        wrong = np.where(pr != gal_y)[0]
        if len(wrong):
            st.markdown("**Misclassified gallery images** (truth → prediction)")
            cols = st.columns(10)
            for j, i in enumerate(wrong[:20]):
                cols[j % 10].image(big(gal_x[i], 80), caption=f"{CLASS_NAMES[gal_y[i]]}→{CLASS_NAMES[pr[i]]}")

st.markdown("---")
st.markdown("<div style='text-align:center;color:#666;font-size:.85rem'>🧠 CIFAR-10 Research Classifier v3.0 · "
            "for research and educational purposes</div>", unsafe_allow_html=True)
