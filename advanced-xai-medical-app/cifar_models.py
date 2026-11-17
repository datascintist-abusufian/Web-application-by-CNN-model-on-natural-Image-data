"""
Model construction for the CIFAR-10 classifier page — shared by app.py and
benchmark/audit_deployed.py so the audit evaluates exactly what the app serves.

Every model is exposed as an xai_utils.Explainable that takes a batch of 32×32 RGB images
in [0, 1]. The transfer models up-sample to 96 px and apply their backbone's own
preprocess_input inside the graph (the same protocol the heads were trained with in
benchmark/train_and_benchmark.py), so gradients and attribution maps are in 32×32 pixel space.
"""
from pathlib import Path

import numpy as np
import tensorflow as tf

import xai_utils as X

ROOT = Path(__file__).resolve().parent
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
TRANSFER_SIZE = 96

SPECS = {
    "CNN": dict(weights=ROOT / "cifar10_cnn.h5", cam="conv2d_1",
                desc="Compact CNN trained from scratch: 2×Conv(32, 64) + MaxPool, Dense(64), softmax"),
    "ResNet50": dict(weights=ROOT / "weights" / "resnet50_head.keras", cam="conv5_block3_out", post=None,
                     desc="Frozen ImageNet ResNet50 (96 px) + trained head GAP→512→256→10"),
    "VGG16": dict(weights=ROOT / "weights" / "vgg16_head.keras", cam="block5_conv3", post="block5_pool",
                  desc="Frozen ImageNet VGG16 (96 px) + trained head GAP→512→256→10"),
    "MobileNetV2": dict(weights=ROOT / "weights" / "mobilenetv2_head.keras", cam="out_relu", post=None,
                        desc="Frozen ImageNet MobileNetV2 (96 px) + trained head GAP→512→256→10"),
}


def _backbone(name):
    from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG16, mobilenet_v2, resnet50, vgg16
    cls, prep = {"ResNet50": (ResNet50, resnet50.preprocess_input),
                 "VGG16": (VGG16, vgg16.preprocess_input),
                 "MobileNetV2": (MobileNetV2, mobilenet_v2.preprocess_input)}[name]
    bb = cls(weights="imagenet", include_top=False, input_shape=(TRANSFER_SIZE, TRANSFER_SIZE, 3))
    bb.trainable = False
    return bb, prep


def load(name):
    """Return (Explainable, info). Raises FileNotFoundError when weights are missing."""
    spec = SPECS[name]
    if not Path(spec["weights"]).exists():
        raise FileNotFoundError(f"{spec['weights'].name} not found")
    if name == "CNN":
        model = tf.keras.models.load_model(spec["weights"], compile=False)
        ex = X.explainable_from_sequential(model, spec["cam"], name=name)
        info = dict(params=int(model.count_params()), input="32×32 RGB, scaled to [0, 1]",
                    weights=spec["weights"].name, desc=spec["desc"], mc_dropout=False)
        return ex, info

    bb, prep = _backbone(name)
    head = tf.keras.models.load_model(spec["weights"], compile=False)
    to_cam = tf.keras.Model(bb.input, bb.get_layer(spec["cam"]).output)
    post = bb.get_layer(spec["post"]) if spec["post"] else None
    head_layers = [l for l in head.layers if not isinstance(l, tf.keras.layers.InputLayer)]

    def feature_fn(x01):
        x = tf.image.resize(tf.convert_to_tensor(x01, tf.float32) * 255.0, (TRANSFER_SIZE, TRANSFER_SIZE))
        return to_cam(prep(x), training=False)

    def _head(a, mc):
        if post is not None:
            a = post(a)
        a = tf.reduce_mean(a, axis=(1, 2))
        return X._run_layers(head_layers, a, mc=mc)

    ex = X.Explainable(feature_fn, lambda a: _head(a, False), lambda a: _head(a, True), name)
    info = dict(params=int(bb.count_params() + head.count_params()),
                input=f"32×32 → {TRANSFER_SIZE}×{TRANSFER_SIZE}, {name} preprocess_input",
                weights=f"ImageNet backbone + {spec['weights'].name}", desc=spec["desc"], mc_dropout=True)
    return ex, info


def to_cifar_input(pil_image):
    """Centre-crop to a square and resize to 32×32 (CIFAR-10 resolution), [0, 1] float."""
    from PIL import Image
    im = pil_image.convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return np.asarray(im.resize((32, 32), Image.BICUBIC)).astype(np.float32) / 255.0
