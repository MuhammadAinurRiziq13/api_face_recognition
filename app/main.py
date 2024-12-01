import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses INFO logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

import tensorflow as tf  
from fastapi import FastAPI, File, UploadFile, Depends,HTTPException,Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Pegawai
from app.face_recognition import FaceRecognition
from app.svm_model import train_face_recognition_model
import numpy as np
import pickle
from joblib import dump  
from mtcnn import MTCNN
import cv2 as cv

# Initialize FastAPI
app = FastAPI()

# Global model and encoder
MODEL_PATH = os.path.join("app", "model", "svm_model.pkl")
ENCODER_PATH = os.path.join("app", "model", "encoder.pkl")
model = None
encoder = None

# Load the model and encoder at startup
@app.on_event("startup")
async def load_model():
    global model, encoder
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        try:
            # Load model
            with open(MODEL_PATH, 'rb') as model_file:
                model = pickle.load(model_file)

            # Load encoder
            with open(ENCODER_PATH, 'rb') as encoder_file:
                encoder = pickle.load(encoder_file)

            print("Model and encoder loaded successfully.")
        except Exception as e:
            print(f"Error loading model or encoder: {e}")
    else:
        print("Model or encoder file not found.")

@app.get("/")
def read_root():
    return {"message": "Hello World"}

# Endpoint for training the model
@app.post("/train_model/")
async def train_model():
    try:
        dataset_directory = "app/dataset"  # Path to dataset
        model, encoder = train_face_recognition_model(dataset_directory)
        await load_model()
        return {"message": "Model and encoder trained and saved successfully."}
    except Exception as e:
        return {"message": f"Error during training: {e}"}


@app.post("/recognize_face/")
async def upload_image(file: UploadFile = File(...)):
    if model is None or encoder is None:
        raise HTTPException(status_code=400, detail="Model and encoder are not loaded properly.")

    # Membaca file image langsung ke dalam memory
    image_bytes = await file.read()

    try:
        face_recognition = FaceRecognition("app/dataset")  
        
        # Ekstraksi wajah
        faces = face_recognition.extract_face_from_bytes(image_bytes)  
        
        if faces is None:
            return {"message": "No face detected in the image."}

        # Extract embedding from the face image
        face_embedding = face_recognition.get_embedding(faces)

        # Ensure the face embedding is the correct shape for the SVM model
        face_embedding = np.expand_dims(face_embedding, axis=0)  # (1, num_features)

        # Perform prediction using the loaded model
        probabilities = model.predict_proba(face_embedding)[0]
        max_prob = np.max(probabilities)
        predicted_label = encoder.inverse_transform([np.argmax(probabilities)])[0]

        # Define a threshold for confidence
        confidence_threshold = 0.4
        if max_prob < confidence_threshold:
            return {"message": "Face detected, but no match found in the database."}
        else:
            return {
                "message": "Face recognized",
                "predicted_name": predicted_label,
                "confidence": f"{max_prob:.2f}"
            }

    except Exception as e:
        return {"message": f"Error during processing: {str(e)}"}


# Endpoint to add a new employee and retrain the model
@app.post("/add_pegawai/")
async def add_pegawai(
    id_pegawai: str = Query(...),  # Mengambil id_pegawai dari query string
    files: list[UploadFile] = File(...),  # Mengambil file dari form-data
    db: Session = Depends(get_db)  # Mendapatkan koneksi DB
):
    # Check if exactly 5 photos are uploaded
    if len(files) != 5:
        return {"message": "You must upload exactly 5 photos."}

    # Create a folder for the employee (id_pegawai) if it does not exist
    dataset_directory = f"app/dataset/{id_pegawai}"
    if not os.path.exists(dataset_directory):
        os.makedirs(dataset_directory)

    # Save the 5 photos and get their file paths
    foto_paths = []
    for idx, file in enumerate(files):
        file_path = f"{dataset_directory}/{idx+1}.jpg"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        foto_paths.append(file_path)

    # Save employee data into the database
    pegawai = Pegawai(
        id_pegawai=id_pegawai,
        foto_1=foto_paths[0] if len(foto_paths) > 0 else None,
        foto_2=foto_paths[1] if len(foto_paths) > 1 else None,
        foto_3=foto_paths[2] if len(foto_paths) > 2 else None,
        foto_4=foto_paths[3] if len(foto_paths) > 3 else None,
        foto_5=foto_paths[4] if len(foto_paths) > 4 else None
    )
    db.add(pegawai)
    db.commit()

    # Retrain the model with the updated dataset
    dataset_directory = "app/dataset"  
    model, encoder = train_face_recognition_model(dataset_directory)
    
    await load_model()

    return {"message": "Employee added successfully, photos saved, and model retrained", "id_pegawai": id_pegawai}