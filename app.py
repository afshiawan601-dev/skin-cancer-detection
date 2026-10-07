import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Skin Cancer Detection", page_icon="🩺", layout="centered")

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("skin_cancer_model.keras", compile=False)
    with open("class_names.json") as f:
        meta = json.load(f)
    return model, meta["class_names"], meta["img_size"]

model, class_names, img_size = load_assets()

st.title("🩺 Skin Cancer Detection")
st.write("Upload a skin lesion image to get a model prediction.")
st.warning("This is an educational demo, not a medical device. It cannot replace a diagnosis by a qualified doctor.")

file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if file is not None:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    x = np.expand_dims(np.asarray(image.resize((img_size, img_size)), dtype="float32"), 0)
    probs = model.predict(x, verbose=0)[0]
    top = int(np.argmax(probs))
    st.subheader(f"Prediction: {class_names[top]}")
    st.write(f"Confidence: {probs[top] * 100:.2f}%")
    st.bar_chart({c: float(p) for c, p in zip(class_names, probs)})
