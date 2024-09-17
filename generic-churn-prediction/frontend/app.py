import streamlit as st
import requests
import pandas as pd

FASTAPI_URL = "http://127.0.0.1:8000"

st.title("Churn Prediction and Explanation System")

# Step 1: Upload dataset and get model summary
st.header("Step 1: Upload Dataset")
uploaded_file = st.file_uploader("Upload dataset for model training", type="csv")

if uploaded_file is not None:
    # Send dataset to backend to upload it and get model info
    files = {"file": uploaded_file.getvalue()}
    response = requests.post(f"{FASTAPI_URL}/upload-dataset/", files={"file": uploaded_file})
    
    if response.status_code == 200:
        model_info = response.json().get("model_info")
        st.write(f"Dataset uploaded successfully. The model to be trained is: {model_info}")
    else:
        st.write("Dataset upload failed.")

# Step 2: Train the model
st.header("Step 2: Train the Model")

if st.button("Train Model"):
    response = requests.post(f"{FASTAPI_URL}/train/")
    
    if response.status_code == 200:
        metrics = response.json().get("metrics")
        st.write("Model trained successfully! Evaluation metrics:")
        st.json(metrics)
    else:
        st.write("Model training failed.")

# Step 3: Predict with the trained model
st.header("Step 3: Predict Using the Trained Model")
uploaded_predict_file = st.file_uploader("Upload data for prediction", type="csv", key="predict")

if uploaded_predict_file is not None and st.button("Predict"):
    files = {"file": uploaded_predict_file.getvalue()}
    response = requests.post(f"{FASTAPI_URL}/predict/", files={"file": uploaded_predict_file})
    
    if response.status_code == 200:
        predictions = response.json().get("predictions")
        st.write("Predictions:")
        st.write(predictions)
    else:
        st.write("Prediction failed.")
