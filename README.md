# CIFAR-10 Image Classifier — Multi-Model Deep Learning Web App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://abusufian-web-application-on-natural-image-data.streamlit.app/)
![Python](https://img.shields.io/badge/python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)

An interactive Streamlit application that classifies natural images from the **CIFAR-10** dataset
(airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) and compares four deep-learning
architectures — a custom **CNN**, **ResNet50**, **VGG16** and **MobileNetV2** — side by side.

**Live demo:** https://abusufian-web-application-on-natural-image-data.streamlit.app/

<p align="center">
  <img src="Real_DL_architect.gif" alt="Deep learning architecture animation" width="600">
</p>

---

## Contents

- [Features](#features)
- [Model comparison results](#model-comparison-results)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Reproducing the benchmark](#reproducing-the-benchmark)
- [Known limitations and roadmap](#known-limitations-and-roadmap)
- [Tech stack](#tech-stack)
- [Author](#author)

---

## Features

The app has two pages, selectable from the sidebar.

### 1. Classifier (`app.py`)

- Choose a model architecture (CNN, ResNet50, VGG16, MobileNetV2) and load it.
- Classify a built-in sample image for any of the 10 classes, or **upload your own image**
  (JPG, PNG, BMP, TIFF).
- Per-class confidence bar chart and sorted-confidence curve.
- Confidence heat-map with **prediction entropy** (raw and normalised), a simple uncertainty measure.
- Top-k predictions table.
- Session metrics: running accuracy, average confidence, inference time and a confusion matrix.

### 2. Model Comparison dashboard (`pages/1_Model_Comparison.py`)

A research-style comparison of all four models trained and evaluated under **one shared protocol**.
Every metric is computed at page load from the files in `results/`. The page does not import
TensorFlow, so it opens instantly.

| Tab | What it shows |
|---|---|
| 🏁 Comparison | Head-to-head table (accuracy with 95% bootstrap CI, macro precision/recall/F1, top-3 accuracy, macro AUC, ECE, NLL, latency, parameters), grouped bar chart, CI plot, auto-generated key findings, CSV download |
| 📈 Learning curves | Train/validation accuracy and loss per epoch for all models on one timeline, best-epoch markers, generalisation-gap curve, convergence-speed curve |
| 🧩 Per-class | Per-class F1/recall/precision profile, model × class heat-map, row-normalised confusion matrices, most-confused class pairs |
| 🎯 Calibration | Reliability diagram, ECE/NLL/Brier, confidence distribution of correct vs. wrong predictions |
| 📉 ROC | One-vs-rest ROC curves (macro-average or per class) with AUC |
| 🛡️ Robustness | Accuracy vs. Gaussian-noise level, and the share of clean accuracy retained |
| ⚡ Efficiency | Accuracy vs. latency with Pareto frontier, throughput, model size, training time |
| 🧪 Significance & ensemble | Exact McNemar tests with Holm correction, error-overlap (Jaccard) matrix, soft-vote ensemble and oracle accuracy |
| 🔎 Deployment audit | Accuracy of the models exactly as `app.py` serves them, with a list of issues found |

---

## Model comparison results

Shared protocol: stratified subsets of CIFAR-10 (seed 42): **10,000 train / 2,000 validation /
5,000 test** images. The CNN is trained from scratch at 32×32. The transfer models use frozen
ImageNet backbones at 96×96 with each backbone's own `preprocess_input` and the same classifier
head (GAP → 512 → 256 → 10). All models use Adam (lr 1e-3), batch 64, up to 40 epochs, and early
stopping on validation loss (patience 8, best weights restored). Hardware: 2-core CPU, no GPU.

| Model | Params | Test accuracy (95% CI) | Macro-F1 | Macro AUC | ECE ↓ | Latency (ms/img, CPU) |
|---|---:|---:|---:|---:|---:|---:|
| **ResNet50** | 24.8 M | **78.6%** (77.5–79.8) | **0.786** | **0.975** | **0.028** | 15.8 |
| MobileNetV2 | 3.0 M | 76.0% (74.8–77.2) | 0.761 | 0.969 | 0.042 | 7.6 |
| VGG16 | 15.1 M | 74.0% (72.8–75.2) | 0.739 | 0.964 | 0.031 | 38.2 |
| CNN (from scratch) | 0.8 M | 67.1% (65.8–68.4) | 0.670 | 0.947 | 0.083 | **2.0** |

Key observations:

- **ResNet50** is the most accurate and best-calibrated model. All pairwise accuracy differences are
  statistically significant (McNemar, Holm-corrected p < 0.05).
- **MobileNetV2** gives the best accuracy-for-cost among the pretrained models, but it is the most
  sensitive to input noise.
- The **CNN** is about 8× faster than ResNet50 and the most robust to noise in relative terms, but
  it overfits the most: the train–validation gap reaches about 20%.
- The models make different mistakes. A **soft-vote ensemble** of all four reaches **83.2%**, and at
  least one model is correct on **93.6%** of test images.
- **Cat** is the hardest class for the three pretrained models. For the CNN, dog is hardest by a
  narrow margin. Cat ↔ dog is the most common confusion for every model.

These numbers come from a reduced, CPU-friendly protocol so that all four models train on identical
data. Full-data training, larger input sizes or fine-tuning would raise every score. See
[Reproducing the benchmark](#reproducing-the-benchmark).

---

## Repository structure

```
.
├── app.py                         # Main Streamlit page: single-image classifier
├── pages/
│   └── 1_Model_Comparison.py      # Model Comparison dashboard (second page)
├── benchmark/
│   ├── train_and_benchmark.py     # Trains and evaluates all 4 models under one protocol
│   ├── audit_deployed.py          # Evaluates the models exactly as app.py serves them
│   └── README.md                  # Benchmark usage notes
├── results/
│   ├── benchmark_summary.json     # Learning curves, latency, robustness, metadata
│   ├── test_predictions.npz       # y_true + softmax probabilities per model
│   └── deployment_audit.json      # Audit of the served models
├── cifar10_cnn.h5                 # Pre-trained CNN weights used by app.py
├── cifar_image_<class>_1.png      # One sample image per class
├── Real_DL_architect.gif          # Architecture animation
├── requirements.txt               # Python dependencies (pinned)
├── packages.txt                   # System packages for Streamlit Cloud
└── .streamlit/config.toml         # Streamlit theme and server settings
```

---

## Getting started

**Requirements:** Python 3.9–3.11 (TensorFlow 2.15 does not support 3.12+).

```bash
git clone https://github.com/datascintist-abusufian/Web-application-by-CNN-model-on-natural-Image-data.git
cd Web-application-by-CNN-model-on-natural-Image-data

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

Then open http://localhost:8501. The **Model Comparison** page appears in the sidebar.

Usage:

1. In the sidebar, select a model and click **Load/Create Model**.
2. Pick a sample class, or upload an image.
3. Click **Analyze Image** to see the prediction, confidence, entropy and top-k classes.
4. Open **Model Comparison** in the sidebar for the four-model analysis.

---

## Reproducing the benchmark

```bash
# Train and evaluate all four models (CPU ≈ 30 min; GPU a few minutes)
python benchmark/train_and_benchmark.py --save-weights

# Evaluate the models exactly as app.py serves them
python benchmark/audit_deployed.py
```

The dashboard picks up the new files in `results/` automatically. On a GPU (e.g. Google Colab) you
can scale up to near-full data and the native 224 px input:

```bash
python benchmark/train_and_benchmark.py --img-size 224 \
  --train-per-class 4500 --val-per-class 500 --test-per-class 1000
```

Useful flags: `--epochs`, `--batch`, `--seed`, `--models CNN,ResNet50`, `--npz <local cifar10.npz>`.
Trained weights go to `results/weights/`, which is git-ignored.

---

## Known limitations and roadmap

Found by `benchmark/audit_deployed.py` and shown in the dashboard's **Deployment audit** tab:

- [ ] **The ResNet50, VGG16 and MobileNetV2 models on the main page are untrained.** No weights file
      exists for them, so `app.py` builds them with a random classifier head and they predict at about
      chance level (≈10–16%). The plan is to ship the trained heads from the benchmark and load them
      in `load_or_create_model()`.
- [ ] **Input preprocessing for the ImageNet backbones.** `preprocess_image()` divides pixels by 255.
      The backbones expect their own `preprocess_input()`.
- [ ] **CNN description mismatch.** `cifar10_cnn.h5` is a compact 2-conv network (167,562
      parameters, 67.0% test accuracy). It differs from `create_cnn_model()` and from the "~1.2M"
      shown in the sidebar.
- [ ] **Uploaded images are scored against the dropdown class**, which skews the session accuracy
      and the confusion matrix. Planned fix: treat uploads as unlabelled.
- [ ] **Results reset when the Top-k slider moves.** Planned fix: keep the last prediction in
      `st.session_state`.
- [ ] Further work: fine-tuning, data augmentation, Grad-CAM explanations, and a Vision Transformer
      baseline.

---

## Tech stack

- **Deep learning:** TensorFlow / Keras (CNN, ResNet50, VGG16, MobileNetV2 transfer learning)
- **App framework:** Streamlit (multi-page)
- **Visualisation:** Plotly
- **Analysis:** NumPy, pandas, scikit-learn, SciPy
- **Dataset:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html): 60,000 32×32 colour
  images in 10 classes (Krizhevsky, 2009)

---

## Author

**Md Abu Sufian** — PhD researcher in Computer Science, University of East London.
Research interests: explainable and multimodal AI for medical and cardiovascular imaging.

- Website: [databizanalyst.com](https://databizanalyst.com)
- GitHub: [@datascintist-abusufian](https://github.com/datascintist-abusufian)

This project is for research and educational purposes. Issues and pull requests are welcome.
