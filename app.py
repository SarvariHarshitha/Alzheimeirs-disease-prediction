import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

st.title(':rainbow[Alzheimers Disease Classification]')
st.write(":gray[This is a simple web app to classify Alzheimers Disease using a Convolutional Neural Network.]")

name = st.text_input("", placeholder="Enter your name")
phone_number = st.text_input("",placeholder ="Enter your phone number")
email = st.text_input("", placeholder = "Enter your email")
file_up = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

def preprocess_image(image, target_size):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image = img_to_array(image)  
    image = image / 255.0      
    image = np.expand_dims(image, axis=0) 
    return image

def predict(image, model):
    processed_image = preprocess_image(image, target_size=(224, 224))
    prediction = model.predict(processed_image)
    label = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    return label, confidence

if file_up is not None: 
    model = load_model("model.h5") 
    image = Image.open(file_up) 
    
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    
    index = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
    try:
        label, confidence = predict(image, model)
        st.write(f"Prediction: :green[{index[label]}]")
    except Exception as e:
        st.write("An error occurred during prediction:")
        st.error(e)

st.markdown('''
 **Created by :blue-background[Harshitha] :tulip:**
''')
