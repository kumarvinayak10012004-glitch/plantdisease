import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("model.keras")

# Class names (15 classes)
class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust',
    'Corn___healthy',
    'Corn___Northern_Leaf_Blight',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___healthy',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)'
]

st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image to detect the disease.")

uploaded_img = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png"])

if uploaded_img is not None:
    img = Image.open(uploaded_img).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

    if st.button("🔍 Predict"):
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        result = class_names[class_index]

        st.success(f"🌱 **Predicted Disease:** {result}")
