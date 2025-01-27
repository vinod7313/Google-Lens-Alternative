import os
import cv2
import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Load CIFAR-10 dataset
def load_cifar10():
    (X_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
    X_train = X_train / 255.0  # Normalize images
    return X_train, y_train.flatten()

# Resize CIFAR-10 images to match ResNet input size
def resize_images(images, size=(224, 224)):
    return np.array([cv2.resize(img, size) for img in images])

# Load the dataset
print("Loading dataset...")
X_train, y_train = load_cifar10()
X_train_resized = resize_images(X_train)

# Approach 1: CNN Feature Extraction
class CNNFeatureExtractor:
    def __init__(self):
        self.model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

    def extract_features(self, images):
        preprocessed_images = preprocess_input(images)
        return self.model.predict(preprocessed_images)

    def find_similar_images(self, query_features, database_features, top_n=5):
        similarities = cosine_similarity(query_features, database_features)
        top_indices = similarities[0].argsort()[-top_n:][::-1]
        return top_indices, similarities[0][top_indices]


# Initialize feature extractor and extract dataset features
print("Extracting CNN features for the dataset...")
cnn_extractor = CNNFeatureExtractor()
database_features = cnn_extractor.extract_features(X_train_resized)

# Flask Routes
@app.route('/')
def index():
    return "Welcome to the Google Lens Alternative API!"

@app.route('/similarity/cnn', methods=['POST'])
def similarity_cnn():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "Invalid file"}), 400

    # Read uploaded image
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)
    img = cv2.imread(file_path)
    img_resized = cv2.resize(img, (224, 224)) / 255.0

    # Extract features
    query_features = cnn_extractor.extract_features(np.expand_dims(img_resized, axis=0))

    # Find similar images
    top_indices, similarities = cnn_extractor.find_similar_images(query_features, database_features)
    similar_images = [{"index": int(idx), "similarity": float(sim)} for idx, sim in zip(top_indices, similarities)]

    # Cleanup uploaded file
    os.remove(file_path)

    return jsonify({"similar_images": similar_images})


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "Invalid file"}), 400

    # Save the file to the "uploads" directory
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    return jsonify({"message": f"File uploaded successfully to {file_path}!"})


# Create upload directory if it doesn't exist
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)

