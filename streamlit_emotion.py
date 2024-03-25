import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

model_path = "emotions.h5"
model = tf.keras.models.load_model(model_path)

class_names = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']

st.title("EMOTION RECOGNITION")

st.write("This project tackles emotion recognition in images using a convolutional neural network (CNN) model, trained on labeled emotional faces. The model predicts the most likely emotion (Anger, Disgust, Fear, Happiness, Sadness, Surprise) displayed in the image.")

st.write("DATASET: [Emotion Dataset](https://huggingface.co/spaces/Pranavadhar/EMOTION/blob/main/emotion_dataset.zip)")
st.write("SAMPLE IMAGES: [Sample Images](https://huggingface.co/spaces/Pranavadhar/EMOTION/blob/main/SAMPLE_IMAGES.zip)")
st.write("GitHub Repository: [GitHub - Pranavadhar](https://github.com/pranavadhar)")
st.write("Portfolio Website: [Pranavadhar - Portfolio](https://pranavadhar.github.io/Pranavadhar-overview---web-work/)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.convert("RGB")
    img_resized = img.resize((128, 128))
    img_array = np.array(img_resized)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]

    st.write(f"Prediction: {predicted_class} (Confidence: {predictions[0][predicted_class_index]:.2f})")

    with st.sidebar:
        st.write("Probabilities:")
        for i, class_name in enumerate(class_names):
            st.write(f"{class_name}: {predictions[0][i]:.4f}")
