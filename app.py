import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, models

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


@st.cache_resource
def build_densenet_model():
    """Build a DenseNet121-based classifier for ocular disorder detection."""
    base_model = DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    )
    # Freeze the base model weights
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize and normalize a fundus image for model input."""
    img = image.convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return img_array


def enhance_contrast(img_array: np.ndarray, contrast: int = 40) -> np.ndarray:
    """Apply contrast enhancement similar to the original preprocessing."""
    img = (img_array * 255).astype(np.uint8)
    alpha_c = 131 * (contrast + 127) / (127 * (131 - contrast))
    gamma_c = 127 * (1 - alpha_c)
    enhanced = cv2.addWeighted(img, alpha_c, img, 0, gamma_c)
    return enhanced.astype(np.float32) / 255.0


# ── Streamlit UI ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="Eye Disease Classifier", layout="centered")

st.title("Automated Eye Disease Detection")
st.markdown(
    "Upload a retinal fundus image and the model will classify it into one of "
    f"**{NUM_CLASSES} ocular disorder categories** using a DenseNet121 backbone."
)

st.sidebar.header("About")
st.sidebar.info(
    "This app uses **DenseNet121** (pre-trained on ImageNet) as a feature "
    "extractor for classifying retinal fundus images. DenseNet's dense "
    "connectivity pattern strengthens feature propagation and is well-suited "
    "for medical imaging tasks where fine-grained features matter."
)

st.sidebar.header("Model Details")
st.sidebar.markdown(
    "- **Architecture:** DenseNet121 + custom head\n"
    "- **Input size:** 120 x 150 px\n"
    f"- **Classes:** {NUM_CLASSES}\n"
    "- **Pre-trained on:** ImageNet"
)

uploaded_file = st.file_uploader(
    "Upload a fundus image", type=["png", "jpg", "jpeg", "bmp", "tiff"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    # Preprocess
    img_array = preprocess_image(image)
    enhanced = enhance_contrast(img_array)

    with col2:
        st.subheader("Enhanced Image")
        st.image(enhanced, use_container_width=True, clamp=True)

    # Predict
    with st.spinner("Loading model and running inference..."):
        model = build_densenet_model()
        input_batch = np.expand_dims(img_array, axis=0)
        predictions = model.predict(input_batch, verbose=0)

    predicted_idx = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_idx])

    st.subheader("Prediction")
    st.metric(label="Predicted Class", value=CLASS_NAMES[predicted_idx])
    st.metric(label="Confidence", value=f"{confidence:.2%}")

    # Show top-5 predictions
    st.subheader("Top 5 Predictions")
    top5_idx = np.argsort(predictions[0])[::-1][:5]
    for rank, idx in enumerate(top5_idx, 1):
        prob = float(predictions[0][idx])
        st.progress(prob, text=f"{rank}. {CLASS_NAMES[idx]} — {prob:.2%}")
else:
    st.info("Please upload a retinal fundus image to get started.")
