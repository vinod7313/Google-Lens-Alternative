# Google-Lens-Alternative
Google Lens Alternative: Image Similarity Search

![image](https://github.com/user-attachments/assets/664055fa-8566-4ab1-8a21-2f821155b984)

Google Lens Alternative: Image Similarity Search
This project implements an alternative to Google Lens by enabling image similarity search using machine learning techniques. The application provides a Flask-based API that identifies and retrieves the most similar images to an input image from a dataset (CIFAR-10).

# Table of Contents
Objective
Features
Approaches
Dataset
Installation
Usage
API Endpoints
Results
Next Steps
References

# Objective: The goal is to develop an image similarity search engine with multiple methods:

Extract image features using a pre-trained CNN (ResNet50).
Provide a scalable solution for real-time image similarity search.
Compare different techniques and evaluate their performance.
Features
Upload an image and retrieve similar images.
Support for feature extraction using CNN-based models.
Lightweight and efficient solution, suitable for real-time applications.
Pretrained ResNet50 as the feature extraction backbone.
Approaches
Convolutional Neural Network (CNN)
Utilizes the ResNet50 architecture pre-trained on ImageNet. Features are extracted from the global average pooling layer, and cosine similarity is used for similarity comparison.
Note: Future extensions will include additional methods like Autoencoders and Siamese Networks.

# Dataset
The project uses the CIFAR-10 dataset for demonstration:
60,000 images (50,000 for training, 10,000 for testing).
10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.
Installation
1. Clone the Repository

git clone https://github.com/yourusername/google-lens-alternative.git
cd google-lens-alternative
2. Install Dependencies

pip install -r requirements.txt
3. Run the Flask Application
python app.py


# API Endpoints
Endpoint	Method	Description
/	GET	Returns a welcome message.
/similarity/cnn	POST	Upload an image to find similar images.
/upload	POST	Upload an image for testing purposes.
Results
Model: ResNet50 (pretrained on ImageNet).
Metric: Cosine similarity for image comparison.
# Performance:
Fast feature extraction with pre-trained ResNet50.
High accuracy in retrieving visually similar images.

