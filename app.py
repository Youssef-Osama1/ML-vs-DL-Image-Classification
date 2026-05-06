import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Dental Image Classifier", page_icon="🦷", layout="centered")

CLASS_NAMES = ["CaS", "CoS", "Gum", "MC", "OC", "OLP", "OT"]
CLASS_DESCRIPTIONS = {
    "CaS": "Calculus",
    "CoS": "Caries",
    "Gum": "Gingivitis",
    "MC" : "Mouth Cancer",
    "OC" : "Oral Cancer",
    "OLP": "Oral Lichen Planus",
    "OT" : "Other",
}

@st.cache_resource
def load_model_from_hf():
    path = hf_hub_download(repo_id="itsYoussefAI/teeth-model", filename="best_resnet.h5")
    return load_model(path)

st.title("🦷 Dental Image Classifier")
st.markdown("---")

uploaded = st.file_uploader("Upload a dental image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded:
    img = Image.open(uploaded)
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, use_container_width=True)

    with col2:
        with st.spinner("Analyzing..."):
            try:
                model = load_model_from_hf()
                arr   = np.expand_dims(resnet_preprocess(
                            np.array(img.convert("RGB").resize((224, 224)), dtype=np.float32)
                        ), axis=0)
                preds     = model.predict(arr, verbose=0)[0]
                top_idx   = int(np.argmax(preds))
                top_class = CLASS_NAMES[top_idx]
                top_conf  = preds[top_idx] * 100
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        st.markdown(f"### {top_class} — {CLASS_DESCRIPTIONS[top_class]}")
        st.metric("Confidence", f"{top_conf:.1f}%")
