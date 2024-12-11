import pickle
from face_recognition import FaceRecognition
import os

def train_face_recognition_model(dataset_directory="dataset"):
    # Load faces and labels
    face_recognition = FaceRecognition(dataset_directory)
    X, Y = face_recognition.load_classes()

    # Train the SVM model
    model = face_recognition.train_model(X, Y)
    
    # Define file paths for the model and encoder
    model_path = os.path.join("model", "svm_model.pkl")
    encoder_path = os.path.join("model", "encoder.pkl")
    
    # Save the model to a separate file
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Save the encoder to a separate file
    with open(encoder_path, 'wb') as f:
        pickle.dump(face_recognition.encoder, f)

    return model, face_recognition.encoder
