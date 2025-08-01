import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.activations import swish
from PIL import Image

# Load model
model = load_model("CrustaScope_model.h5", custom_objects={"swish": swish})

st.title("🦐 WSSV Detection for Shrimps")
st.write("Upload an image to detect if the shrimp is **Healthy** or has **WSSV**.")

uploaded_file = st.file_uploader("Upload Shrimp Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Shrimp Image", use_column_width=True)

    # Preprocess
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array, verbose=0)[0][0]
    label = "WSSV" if prediction > 0.5 else "HEALTHY"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {round(confidence*100, 2)}%")
