import json
import os
import streamlit as st
import numpy as np
import cv2
import time
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import (
    DenseNet121,
    EfficientNetB3,
    EfficientNetV2S,
    ConvNeXtTiny,
    MobileNetV2,
)
from tensorflow.keras import layers, models

# ── Custom ResNet18 (built from scratch) ─────────────────────────────────────

def _residual_block(x, filters, stride=1):
    """A single residual block with two 3×3 convolutions and a skip connection."""
    shortcut = x

    x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # Projection shortcut when dimensions change
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


def build_custom_resnet18(input_shape):
    """Build a ResNet-18 from scratch (no pretrained weights).

    Architecture: conv7×7 → [2,2,2,2] residual blocks → GlobalAvgPool
    Total ~11M parameters with the classification head.
    """
    inputs = layers.Input(shape=input_shape)

    # Initial convolution (stem)
    x = layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    # Residual stages: [64, 128, 256, 512] × 2 blocks each
    for filters, stride in [(64, 1), (64, 1)]:
        x = _residual_block(x, 64, stride)
    for i, stride in enumerate([2, 1]):
        x = _residual_block(x, 128, stride)
    for i, stride in enumerate([2, 1]):
        x = _residual_block(x, 256, stride)
    for i, stride in enumerate([2, 1]):
        x = _residual_block(x, 512, stride)

    x = layers.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inputs, x, name="custom_resnet18")


# 26 ocular disorder classes (sorted alphabetically to match LabelEncoder ordering)
CLASS_NAMES = [
    "AMD", "Bleeding", "Blur Fundus", "Cataract", "Coats", "Cotton Wool Spots",
    "Diabetic Retinopathy", "Drusen", "Glaucoma", "Hemorrhage",
    "Healthy", "Hypertensive Retinopathy", "Laser Scars", "Macular Hole",
    "Maculopathy", "Myopia", "Normal Fundus", "Optic Disc Pallor",
    "Preretinal Hemorrhage", "Proliferative DR", "Retinal Detachment",
    "Retinitis Pigmentosa", "Retinoblastoma", "STARE Normal",
    "Toxoplasmosis", "Vessel Tortuosity",
]

NUM_CLASSES = len(CLASS_NAMES)
IMG_HEIGHT, IMG_WIDTH = 120, 150

# ── Model registry ───────────────────────────────────────────────────────────

MODEL_INFO = {
    "DenseNet121": {
        "builder": DenseNet121,
        "year": 2017,
        "params": "~7M base",
        "description": (
            "Dense connectivity where each layer receives features from all "
            "preceding layers. Excellent feature reuse and gradient flow — "
            "widely used in medical imaging (e.g. CheXNet)."
        ),
        "strengths": "Feature reuse, strong on small datasets, parameter efficient",
        "input_note": None,
    },
    "EfficientNetB3": {
        "builder": EfficientNetB3,
        "year": 2019,
        "params": "~12M base",
        "description": (
            "Uses compound scaling to balance network depth, width, and "
            "resolution. Top performer on fundus classification benchmarks "
            "with excellent accuracy-to-compute ratio."
        ),
        "strengths": "Best accuracy/FLOP ratio, compound scaling, state-of-the-art",
        "input_note": None,
    },
    "EfficientNetV2S": {
        "builder": EfficientNetV2S,
        "year": 2021,
        "params": "~21M base",
        "description": (
            "Successor to EfficientNet with fused convolutions in early "
            "layers and progressive learning. Trains 5-11x faster than V1 "
            "with better accuracy on medical imaging benchmarks."
        ),
        "strengths": "Faster training, improved accuracy over V1, progressive learning",
        "input_note": None,
    },
    "ConvNeXtTiny": {
        "builder": ConvNeXtTiny,
        "year": 2022,
        "params": "~28M base",
        "description": (
            "A modernized pure-CNN that incorporates Transformer-era design "
            "choices (large kernels, LayerNorm, GELU). Matches or beats "
            "Vision Transformers while remaining fully convolutional. "
            "Achieved 97.96% on retinal OCT classification."
        ),
        "strengths": "State-of-the-art CNN, Transformer-level accuracy, modern design",
        "input_note": None,
    },
    "MobileNetV2": {
        "builder": MobileNetV2,
        "year": 2018,
        "params": "~3.4M base",
        "description": (
            "Depthwise separable convolutions with inverted residuals for "
            "lightweight inference. Ideal for deploying retinal screening "
            "on mobile or edge devices in clinical settings."
        ),
        "strengths": "Ultra-lightweight, fast inference, mobile-friendly",
        "input_note": None,
    },
    "ResNet18 (Custom)": {
        "builder": None,  # Built from scratch — see build_custom_resnet18()
        "year": 2015,
        "params": "~11M",
        "description": (
            "Hand-coded ResNet-18 built entirely from scratch using Keras "
            "layers — no pretrained weights. Implements the original "
            "residual-learning framework (He et al., 2015) with skip "
            "connections that enable training of deeper networks. This is "
            "our own implementation, not imported from tf.keras.applications."
        ),
        "strengths": "Custom-built, interpretable architecture, residual learning pioneer",
        "input_note": "custom",
    },
}

# ── PyTorch models (pre-computed results from Colab) ────────────────────────

PYTORCH_MODEL_INFO = {
    "ConvNeXtV2-Tiny": {
        "year": 2023,
        "params": "~28.6M base",
        "description": (
            "Successor to ConvNeXt with Global Response Normalization (GRN) "
            "and MAE-based self-supervised pretraining. Best single model on "
            "multi-label retinal classification benchmarks (AUC 0.9967)."
        ),
        "strengths": "GRN layers, self-supervised pretraining, top retinal benchmark results",
        "timm_id": "convnextv2_tiny.fcmae_ft_in22k_in1k",
    },
    "SwinV2-Tiny": {
        "year": 2022,
        "params": "~28M base",
        "description": (
            "Hierarchical Vision Transformer with shifted-window attention "
            "and log-spaced continuous position bias. Best mean rank across "
            "all retinal tasks in a 2025 systematic evaluation."
        ),
        "strengths": "Best consistency across retinal tasks, hierarchical features, scalable",
        "timm_id": "swinv2_tiny_window8_256",
    },
    "MaxViT-Tiny": {
        "year": 2022,
        "params": "~31M base",
        "description": (
            "Multi-axis Vision Transformer combining blocked local attention "
            "and dilated global attention with MBConv. Used in MaxGlaViT "
            "(2025) for glaucoma staging from fundus images."
        ),
        "strengths": "Local + global attention, efficient multi-scale, strong on fundus",
        "timm_id": "maxvit_tiny_tf_224.in1k",
    },
}

PYTORCH_RESULTS_PATH = os.path.join(os.path.dirname(__file__), "pytorch_results.json")
TF_RESULTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "tf_results.json")
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")


def load_pytorch_results() -> dict | None:
    """Load pre-computed PyTorch model results from JSON."""
    if os.path.exists(PYTORCH_RESULTS_PATH):
        with open(PYTORCH_RESULTS_PATH) as f:
            return json.load(f)
    return None


def load_tf_results() -> dict | None:
    """Load pre-computed TensorFlow training results from JSON."""
    if os.path.exists(TF_RESULTS_PATH):
        with open(TF_RESULTS_PATH) as f:
            return json.load(f)
    return None


def _weight_path(model_name: str) -> str:
    """Return the path to saved .keras weights for a model."""
    safe = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    return os.path.join(WEIGHTS_DIR, f"{safe}.keras")


def has_trained_weights(model_name: str) -> bool:
    """Check if trained weights exist for a model."""
    return os.path.exists(_weight_path(model_name))


@st.cache_resource
def load_model(model_name: str):
    """Load a trained model from saved weights."""
    path = _weight_path(model_name)
    model = tf.keras.models.load_model(path)
    return model


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize and normalize a fundus image for model input."""
    img = image.convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    return np.array(img, dtype=np.float32) / 255.0


def enhance_contrast(img_array: np.ndarray, contrast: int = 40) -> np.ndarray:
    """Apply contrast enhancement similar to the original preprocessing."""
    img = (img_array * 255).astype(np.uint8)
    alpha_c = 131 * (contrast + 127) / (127 * (131 - contrast))
    gamma_c = 127 * (1 - alpha_c)
    enhanced = cv2.addWeighted(img, alpha_c, img, 0, gamma_c)
    return enhanced.astype(np.float32) / 255.0


def run_inference(model_name: str, input_batch: np.ndarray) -> dict:
    """Run inference and return predictions with timing info."""
    model = load_model(model_name)
    # Warm-up run (first call has overhead)
    model.predict(input_batch, verbose=0)
    # Timed run
    start = time.perf_counter()
    predictions = model.predict(input_batch, verbose=0)
    elapsed = time.perf_counter() - start
    return {"predictions": predictions[0], "time_ms": elapsed * 1000}


# ── Streamlit UI ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="Eye Disease Classifier", layout="wide")

st.title("Automated Eye Disease Detection — Model Comparison")
st.markdown(
    "Upload a retinal fundus image and compare how **6 TensorFlow architectures** "
    "(including a **custom-built ResNet18**) "
    f"classify it across **{NUM_CLASSES} ocular disorder categories**, plus "
    "**3 PyTorch models** with pre-computed benchmark results from Colab."
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.header("Select Models to Compare")
selected_models = []
models_with_weights = [name for name in MODEL_INFO if has_trained_weights(name)]
models_without_weights = [name for name in MODEL_INFO if not has_trained_weights(name)]

if not models_with_weights:
    st.sidebar.error(
        "No trained weights found! Run `tf_fundus_train.ipynb` on Google Colab "
        "and place the `weights/` folder in the project root."
    )

for name in models_with_weights:
    if st.sidebar.checkbox(name, value=(name in ("DenseNet121", "EfficientNetV2S", "ConvNeXtTiny"))):
        selected_models.append(name)

if models_without_weights:
    st.sidebar.caption(
        f"Missing weights for: {', '.join(models_without_weights)}. "
        "Train them in Colab to enable."
    )

if not selected_models and models_with_weights:
    st.sidebar.warning("Select at least one model.")

st.sidebar.divider()
st.sidebar.header("Model Reference — TensorFlow (Live)")
for name, info in MODEL_INFO.items():
    with st.sidebar.expander(f"{name} ({info['year']})"):
        st.markdown(f"**Year:** {info['year']}  |  **Parameters:** {info['params']}")
        st.markdown(f"**Strengths:** {info['strengths']}")
        st.markdown(info["description"])

st.sidebar.divider()
st.sidebar.header("Model Reference — PyTorch (Colab)")
for name, info in PYTORCH_MODEL_INFO.items():
    with st.sidebar.expander(f"{name} ({info['year']})"):
        st.markdown(f"**Year:** {info['year']}  |  **Parameters:** {info['params']}")
        st.markdown(f"**Strengths:** {info['strengths']}")
        st.markdown(info["description"])
        st.caption(f"`timm` model ID: `{info['timm_id']}`")

# ── Main content ─────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload a fundus image", type=["png", "jpg", "jpeg", "bmp", "tiff"]
)

if uploaded_file is not None and selected_models:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    img_array = preprocess_image(image)
    enhanced = enhance_contrast(img_array)
    with col2:
        st.subheader("Enhanced Image")
        st.image(enhanced, use_container_width=True, clamp=True)

    st.divider()

    # ── Run all selected models ──────────────────────────────────────────────
    input_batch = np.expand_dims(img_array, axis=0)
    results = {}

    progress_bar = st.progress(0, text="Running models...")
    for i, model_name in enumerate(selected_models):
        progress_bar.progress(
            (i) / len(selected_models),
            text=f"Running {model_name}...",
        )
        results[model_name] = run_inference(model_name, input_batch)
    progress_bar.progress(1.0, text="All models complete!")

    # ── Side-by-side predictions ─────────────────────────────────────────────
    st.subheader("Predictions by Model")

    cols = st.columns(len(selected_models))
    for col, model_name in zip(cols, selected_models):
        preds = results[model_name]["predictions"]
        pred_idx = int(np.argmax(preds))
        conf = float(preds[pred_idx])
        elapsed = results[model_name]["time_ms"]

        with col:
            st.markdown(f"### {model_name}")
            st.metric("Predicted Class", CLASS_NAMES[pred_idx])
            st.metric("Confidence", f"{conf:.2%}")
            st.metric("Inference Time", f"{elapsed:.1f} ms")

            st.markdown("**Top 5:**")
            top5 = np.argsort(preds)[::-1][:5]
            for rank, idx in enumerate(top5, 1):
                prob = float(preds[idx])
                st.progress(prob, text=f"{rank}. {CLASS_NAMES[idx]} — {prob:.2%}")

    # ── Comparison table ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("Comparison Summary")

    summary_data = []
    for model_name in selected_models:
        preds = results[model_name]["predictions"]
        pred_idx = int(np.argmax(preds))
        conf = float(preds[pred_idx])
        top3 = np.argsort(preds)[::-1][:3]
        summary_data.append({
            "Model": model_name,
            "Year": MODEL_INFO[model_name]["year"],
            "Prediction": CLASS_NAMES[pred_idx],
            "Confidence": f"{conf:.2%}",
            "Inference (ms)": f"{results[model_name]['time_ms']:.1f}",
            "Top-3": ", ".join(CLASS_NAMES[i] for i in top3),
            "Parameters": MODEL_INFO[model_name]["params"],
        })

    st.table(summary_data)

    # ── Agreement analysis ───────────────────────────────────────────────────
    if len(selected_models) > 1:
        st.subheader("Model Agreement")
        all_preds = [
            CLASS_NAMES[int(np.argmax(results[m]["predictions"]))]
            for m in selected_models
        ]
        unique_preds = set(all_preds)
        if len(unique_preds) == 1:
            st.success(
                f"All {len(selected_models)} models agree: **{all_preds[0]}**"
            )
        else:
            st.warning(
                f"Models disagree — {len(unique_preds)} distinct predictions: "
                f"{', '.join(f'**{p}**' for p in unique_preds)}"
            )
            st.markdown(
                "Disagreement may indicate an ambiguous or challenging image. "
                "Consider the consensus of the majority or the model with "
                "highest confidence."
            )

    # ── TF benchmark results (from Colab training) ─────────────────────────
    tf_results = load_tf_results()
    if tf_results:
        st.divider()
        st.subheader("TensorFlow Models — Training Benchmark Results")
        st.caption("Test-set metrics from Colab training.")
        tf_bench_cols = st.columns(min(len(tf_results), 3))
        tf_items = list(tf_results.items())
        for i, (mname, mres) in enumerate(tf_items):
            with tf_bench_cols[i % len(tf_bench_cols)]:
                st.markdown(f"**{mname}**")
                st.metric("Test Accuracy", f"{mres['test_accuracy']:.2%}")
                st.metric("Inference", f"{mres['inference_ms']:.1f} ms")
                st.caption(f"Trained {mres['epochs_trained']} epochs, best val acc {mres['best_val_accuracy']:.2%}")

    # ── PyTorch model results (pre-computed from Colab) ─────────────────────
    st.divider()
    st.subheader("PyTorch Models — Pre-computed Results from Colab")

    pytorch_results = load_pytorch_results()
    if pytorch_results is None:
        st.info(
            "No PyTorch results found. Run the `pytorch_fundus_eval.ipynb` "
            "notebook on Google Colab to generate `pytorch_results.json`, "
            "then place it in the project root."
        )
    else:
        st.markdown(
            "These results were generated on **Google Colab** using PyTorch + "
            "`timm`. They show benchmark metrics on the test set — not live "
            "inference on the uploaded image."
        )

        pt_cols = st.columns(len(PYTORCH_MODEL_INFO))
        for col, (model_name, info) in zip(pt_cols, PYTORCH_MODEL_INFO.items()):
            model_results = pytorch_results.get(model_name, {})
            with col:
                st.markdown(f"### {model_name}")
                st.caption(f"{info['year']} · {info['params']}")
                accuracy = model_results.get("accuracy", "N/A")
                f1 = model_results.get("f1_weighted", "N/A")
                auc = model_results.get("auc_macro", "N/A")
                inf_ms = model_results.get("inference_ms", "N/A")

                if isinstance(accuracy, (int, float)):
                    st.metric("Test Accuracy", f"{accuracy:.2%}")
                else:
                    st.metric("Test Accuracy", accuracy)
                if isinstance(f1, (int, float)):
                    st.metric("Weighted F1", f"{f1:.4f}")
                else:
                    st.metric("Weighted F1", f1)
                if isinstance(auc, (int, float)):
                    st.metric("Macro AUC", f"{auc:.4f}")
                else:
                    st.metric("Macro AUC", auc)
                if isinstance(inf_ms, (int, float)):
                    st.metric("Inference Time", f"{inf_ms:.1f} ms")
                else:
                    st.metric("Inference Time", inf_ms)

        # Comparison table for PyTorch models
        pt_summary = []
        for model_name, info in PYTORCH_MODEL_INFO.items():
            r = pytorch_results.get(model_name, {})
            acc = r.get("accuracy", "N/A")
            pt_summary.append({
                "Model": model_name,
                "Framework": "PyTorch",
                "Year": info["year"],
                "Parameters": info["params"],
                "Accuracy": f"{acc:.2%}" if isinstance(acc, (int, float)) else acc,
                "F1 (weighted)": f"{r.get('f1_weighted', 'N/A'):.4f}" if isinstance(r.get("f1_weighted"), (int, float)) else "N/A",
                "AUC (macro)": f"{r.get('auc_macro', 'N/A'):.4f}" if isinstance(r.get("auc_macro"), (int, float)) else "N/A",
            })
        st.table(pt_summary)

elif uploaded_file is None:
    if not models_with_weights:
        st.warning(
            "**Setup required:** Run `tf_fundus_train.ipynb` on Google Colab to train the models, "
            "then upload the `weights/` folder to this project. Without trained weights, "
            "live predictions cannot be made."
        )
    else:
        st.info("Please upload a retinal fundus image to get started.")
