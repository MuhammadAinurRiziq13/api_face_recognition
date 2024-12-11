import cv2 as cv
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os

class FaceRecognition:
    def __init__(self, dataset_directory="dataset"):
        self.directory = dataset_directory  # Mengarah ke app/dataset
        self.target_size = (160, 160)
        self.detector = MTCNN()
        self.embedder = FaceNet()
        self.encoder = LabelEncoder()
    
    def extract_face(self, filename):
        """ Extract face from image using MTCNN detector """
        img = cv.imread(filename)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(img)
        if len(faces) == 0:
            return None
        x, y, w, h = faces[0]['box']
        x, y = abs(x), abs(y)
        face = img[y:y+h, x:x+w]
        face_arr = cv.resize(face, self.target_size)
        return face_arr
    
    def extract_face_from_bytes(self, image_bytes):
        """ Extract face from image using MTCNN detector """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv.imdecode(nparr, cv.IMREAD_COLOR)
        if img is None:
            return None

        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(img)
        if len(faces) == 0:
            return None
        x, y, w, h = faces[0]['box']
        x, y = abs(x), abs(y)
        face = img[y:y+h, x:x+w]
        face_arr = cv.resize(face, self.target_size)
        return face_arr

    def load_faces(self, dir):
        """ Load faces from a directory """
        faces = []
        for im_name in os.listdir(dir):
            path = os.path.join(dir, im_name)
            face = self.extract_face(path)
            if face is not None:
                faces.append(face)
        return faces

    def load_classes(self):
        """ Load classes and faces """
        X, Y = [], []
        for sub_dir in os.listdir(self.directory):
            path = os.path.join(self.directory, sub_dir)
            faces = self.load_faces(path)
            labels = [sub_dir for _ in range(len(faces))]
            X.extend(faces)
            Y.extend(labels)
        return np.asarray(X), np.asarray(Y)

    def get_embedding(self, face_img):
        """ Get face embedding from FaceNet model """
        face_img = face_img.astype('float32')
        face_img = np.expand_dims(face_img, axis=0)
        return self.embedder.embeddings(face_img)[0]

    def train_model(self, X, Y):
        """ Train the SVM model """
        X_embedded = np.array([self.get_embedding(face) for face in X])
        Y_encoded = self.encoder.fit_transform(Y)
        model = SVC(kernel='linear', probability=True)
        model.fit(X_embedded, Y_encoded)
        return model

    def predict(self, face_embedding):
        """ Predict the label of a face using trained SVM model with confidence score """
        # Make prediction using the trained model
        prediction = self.model.predict([face_embedding])
        predicted_label = self.encoder.inverse_transform(prediction)[0]

        # Get confidence score (probability) for the prediction
        prediction_confidence = self.model.decision_function([face_embedding])[0]
        confidence_score = np.max(np.abs(prediction_confidence))  # Take the maximum confidence score
        
        return predicted_label, confidence_score

    def evaluate(self, X, Y):
        """ Evaluate the model accuracy """
        X_embedded = np.array([self.get_embedding(face) for face in X])
        Y_encoded = self.encoder.transform(Y)
        predictions = self.model.predict(X_embedded)
        return accuracy_score(Y_encoded, predictions)
