from fastapi import FastAPI, UploadFile, File
from data_utils import preprocess_data, load_data, train_test_data
from model import train_model, evaluate_model, load_model, save_model
import os

app = FastAPI()

DATA_DIR = "data/"
MODEL_DIR = "models/"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Store the uploaded dataset globally to access it later
uploaded_dataset = None

@app.post("/upload-dataset/")
async def upload_dataset(file: UploadFile = File(...)):
    global uploaded_dataset
    file_path = f"{DATA_DIR}{file.filename}"
    
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())
    
    uploaded_dataset = file_path
    
    return {"message": "Dataset uploaded successfully", "model_info": "Random Forest"}

@app.post("/train/")
async def train():
    global uploaded_dataset
    if uploaded_dataset is None:
        return {"error": "No dataset uploaded"}

    X_train, X_test, y_train, y_test = train_test_data(uploaded_dataset)
    model = train_model(X_train, y_train)
    
    save_model(model, MODEL_DIR + "churn_model.joblib")
    
    evaluation_metrics = evaluate_model(model, X_test, y_test)
    return {"message": "Model trained successfully", "metrics": evaluation_metrics}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    model = load_model(MODEL_DIR + "churn_model.joblib")
    data = load_data(file.file)
    X, _ = preprocess_data(data)
    
    predictions = model.predict(X)
    return {"predictions": predictions.tolist()}
