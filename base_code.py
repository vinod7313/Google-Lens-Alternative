# Import Libraries
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Lambda
)
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score


# Dataset Loader
def load_cifar10():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0
    y_train, y_test = y_train.flatten(), y_test.flatten()
    return X_train, X_test, y_train, y_test


# Split Dataset
def split_dataset(images, labels, test_size=0.2):
    return train_test_split(images, labels, test_size=test_size, random_state=42)


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


# Approach 2: Autoencoder
def build_autoencoder(img_size):
    input_img = Input(shape=(img_size[0], img_size[1], 3))

    # Encoder
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(input_img)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    encoded = MaxPooling2D((2, 2), padding="same")(x)

    # Decoder
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder, encoder


# Approach 3: Siamese Network
def build_siamese_network(input_shape):
    def build_base_network(input_shape):
        input_img = Input(shape=input_shape)
        x = Conv2D(64, (3, 3), activation="relu", padding="same")(input_img)
        x = MaxPooling2D((2, 2), padding="same")(x)
        x = Flatten()(x)
        x = Dense(128, activation="relu")(x)
        return Model(input_img, x)

    base_network = build_base_network(input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([processed_a, processed_b])
    outputs = Dense(1, activation="sigmoid")(distance)

    siamese_model = Model([input_a, input_b], outputs)
    siamese_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return siamese_model


# Evaluation Metrics
def evaluate_model(y_true, y_pred, threshold=0.5):
    y_pred_binary = (y_pred >= threshold).astype(int)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    accuracy = accuracy_score(y_true, y_pred_binary)
    return {"precision": precision, "recall": recall, "accuracy": accuracy}


# Main Script
if __name__ == "__main__":
    # Load CIFAR-10 dataset
    X_train, X_test, y_train, y_test = load_cifar10()

    # Resize for consistent input size (ResNet expects 224x224)
    X_train_resized = np.array([cv2.resize(img, (224, 224)) for img in X_train])
    X_test_resized = np.array([cv2.resize(img, (224, 224)) for img in X_test])

    # CNN Feature Extraction
    print("Extracting features using CNN...")
    cnn_extractor = CNNFeatureExtractor()
    train_features = cnn_extractor.extract_features(X_train_resized)
    test_features = cnn_extractor.extract_features(X_test_resized)
    print("CNN Features Extracted!")

    # Autoencoder
    print("Training Autoencoder...")
    autoencoder, encoder = build_autoencoder((32, 32))
    autoencoder.fit(X_train, X_train, epochs=10, batch_size=32, validation_split=0.2)
    print("Autoencoder Trained!")

    # Siamese Network
    print("Training Siamese Network...")
    siamese_model = build_siamese_network((32, 32, 3))
    siamese_model.summary()

    # Evaluation
    print("Evaluating CNN approach...")
    query_img = test_features[:1]  # Use first test image as query
    top_indices, similarities = cnn_extractor.find_similar_images(query_img, train_features)
    print(f"Top indices: {top_indices}, Similarities: {similarities}")

    print("Similarity search complete!")
