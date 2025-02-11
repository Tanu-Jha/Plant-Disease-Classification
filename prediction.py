import os
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import openai

# --- Page Configuration ---
st.set_page_config(page_title="Plant Disease Classifier", layout="wide")

# --- OpenAI API Key ---
# It is best to store your API key securely (e.g., in Streamlit secrets).
openai.api_key = st.secrets["api_key"]

# --- Load the Model (with caching for performance) ---
@st.cache_resource(show_spinner=True)
def load_model():
    return tf.keras.models.load_model("vgg_model_complete.h5")

model = load_model()

# --- Define Class Indices and Labels ---
class_indices = {
    'Apple___Apple_scab': 0, 
    'Apple___Black_rot': 1, 
    'Apple___Cedar_apple_rust': 2, 
    'Apple___healthy': 3, 
    'Background_without_leaves': 4, 
    'Blueberry___healthy': 5, 
    'Cherry___Powdery_mildew': 6, 
    'Cherry___healthy': 7, 
    'Corn___Cercospora_leaf_spot Gray_leaf_spot': 8, 
    'Corn___Common_rust': 9, 
    'Corn___Northern_Leaf_Blight': 10, 
    'Corn___healthy': 11, 
    'Grape___Black_rot': 12, 
    'Grape___Esca_(Black_Measles)': 13, 
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 14, 
    'Grape___healthy': 15, 
    'Orange___Haunglongbing_(Citrus_greening)': 16, 
    'Peach___Bacterial_spot': 17, 
    'Peach___healthy': 18, 
    'Pepper,_bell___Bacterial_spot': 19, 
    'Pepper,_bell___healthy': 20, 
    'Potato___Early_blight': 21, 
    'Potato___Late_blight': 22, 
    'Potato___healthy': 23, 
    'Raspberry___healthy': 24, 
    'Soybean___healthy': 25, 
    'Squash___Powdery_mildew': 26, 
    'Strawberry___Leaf_scorch': 27, 
    'Strawberry___healthy': 28, 
    'Tomato___Bacterial_spot': 29, 
    'Tomato___Early_blight': 30, 
    'Tomato___Late_blight': 31, 
    'Tomato___Leaf_Mold': 32, 
    'Tomato___Septoria_leaf_spot': 33, 
    'Tomato___Spider_mites Two-spotted_spider_mite': 34, 
    'Tomato___Target_Spot': 35, 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 36, 
    'Tomato___Tomato_mosaic_virus': 37, 
    'Tomato___healthy': 38
}

# Create a list of class labels ordered by their index
class_labels = [None] * len(class_indices)
for label, idx in class_indices.items():
    class_labels[idx] = label

# --- Function to Fetch Detailed Disease Information ---
def get_disease_details(disease):
    prompt = (
        f"Provide a brief summary of {disease} in 5-6 short sentences. "
        "If the plant is healthy, just confirm that. "
        "Otherwise, include key symptoms and one main treatment option."
    )
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # Change to your available model if needed.
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error fetching details: {e}"

# --- Main Interface ---
st.title("🌿 Plant Disease Classifier")
st.write(
    """
    Upload an image of a plant leaf below to identify its disease and receive detailed information about it.
    """
)

# --- Sidebar Instructions ---
with st.sidebar:
    st.header("How It Works")
    st.markdown(
        """
        1. **Upload an Image:** Choose a clear image of a plant leaf (jpg, jpeg, or png).
        2. **Prediction:** The app will analyze the image and display the prediction.
        3. **Details:** Get comprehensive information about the disease including symptoms, causes, and treatment options.
        """
    )
    st.info("If the leaf is healthy, the system will confirm that no treatment is necessary.")

# --- Image Upload ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Create a two-column layout: left for image, right for prediction & details.
    col1, col2 = st.columns(2)

    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        # Preprocess the image: resize and normalize pixel values.
        image_array = np.array(image)
        image_resized = cv2.resize(image_array, (224, 224))
        image_normalized = image_resized / 255.0
        image_input = np.expand_dims(image_normalized, axis=0)

    with col2:
        if st.button("Predict"):
            with st.spinner("Analyzing the image..."):
                prediction = model.predict(image_input)
                predicted_index = np.argmax(prediction, axis=1)[0]
                predicted_class = class_labels[predicted_index]
            st.success(f"**Prediction:** {predicted_class}")
            
            with st.spinner("Retrieving detailed information..."):
                details = get_disease_details(predicted_class)
            st.markdown("### Detailed Information")
            st.write(details)
