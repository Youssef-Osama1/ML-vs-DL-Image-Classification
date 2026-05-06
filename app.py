import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from huggingface_hub import hf_hub_download
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dental Image Classifier",
    page_icon="🦷",
    layout="centered",
)

# ── Class names (must match folder order used during training) ─────────────────
CLASS_NAMES = ["CaS", "CoS", "Gum", "MC", "OC", "OLP", "OT"]

CLASS_DESCRIPTIONS = {
    "CaS" : "Calculus",
    "CoS" : "Caries",
    "Gum" : "Gingivitis",
    "MC"  : "Mouth Cancer",
    "OC"  : "Oral Cancer",
    "OLP" : "Oral Lichen Planus",
    "OT"  : "Other",
}

IMG_SIZE = (224, 224)

# ── Model loader (cached so it only loads once) ────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    HF_REPO = "itsYoussefAI/teeth-model"

    try:
        with st.spinner("Downloading model weights... (first run only)"):
            model_path = hf_hub_download(
                repo_id=HF_REPO,
                filename="best_resnet.h5",
            )
        models["ResNet50"] = load_model(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        models["ResNet50"] = None

    return models

# ── Preprocessing helpers ──────────────────────────────────────────────────────
def preprocess(img: Image.Image, model_name: str) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    if model_name == "MobileNetV2":
        arr = mobilenet_preprocess(arr)
    else:
        arr = resnet_preprocess(arr)
    return arr

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🦷 Dental Image Classifier")
st.markdown(
    "Upload a dental image and get an instant prediction using our "
    "pre-trained deep learning models."
)
st.markdown("---")

# Upload
uploaded = st.file_uploader(
    "Upload a dental image",
    type=["jpg", "jpeg", "png", "bmp"],
    label_visibility="collapsed",
)

if uploaded:
    img = Image.open(uploaded)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)

    with col2:
        models = load_models()
        model = models.get("ResNet50")

        if model is None:
            st.error("Model file `best_resnet.h5` not found. Make sure it is in the same folder as `app.py`.")
        else:
            with st.spinner("Analyzing..."):
                x     = preprocess(img, "ResNet50")
                preds = model.predict(x, verbose=0)[0]

            top_idx   = int(np.argmax(preds))
            top_class = CLASS_NAMES[top_idx]
            top_conf  = float(preds[top_idx]) * 100
            full_name = CLASS_DESCRIPTIONS.get(top_class, top_class)

            st.success(f"**{top_class}** — {full_name}")
            st.metric("Confidence", f"{top_conf:.1f}%")

            # Top-3 bar chart
            top3_idx  = np.argsort(preds)[::-1][:3]
            top3_conf = [preds[i] * 100 for i in top3_idx]
            top3_lbls = [CLASS_NAMES[i] for i in top3_idx]

            st.markdown("**Top-3 Predictions**")
            for lbl, conf in zip(top3_lbls, top3_conf):
                bar_fill = int(conf / 100 * 20)
                bar = "█" * bar_fill + "░" * (20 - bar_fill)
                st.markdown(
                    f"`{lbl}` &nbsp; {bar} &nbsp; **{conf:.1f}%**",
                    unsafe_allow_html=True,
                )

else:
    st.info("⬆️ Upload an image above to get started.")