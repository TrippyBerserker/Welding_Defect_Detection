# Welding Defect Detection using ANN + PyTorch

Project Overview

This project aims to detect welding defects using Artificial Neural Networks (ANN) implemented in PyTorch. The model is trained to classify images as either containing a defect or not. The system is integrated into a Flask web application, allowing users to upload images for real-time defect detection.

Features

Deep Learning Model: Utilizes an ANN for defect classification.
PyTorch Implementation: Built and trained using PyTorch.
Flask Web Application: Provides an intuitive interface for users to upload images.
MongoDB Integration: Supports user authentication with login/signup features.
Image Upload & Prediction: Users can upload images to receive defect detection results.

Dataset & Limitations

The dataset consists of welding images labeled as 'defect' and 'no defect.'

Limitations:

Limited Dataset Size: A small number of images may lead to:
Overfitting: The model might memorize the training data instead of generalizing.
Underfitting: Insufficient data may prevent the model from learning meaningful patterns.
Data Imbalance: If one class (defect/no defect) has significantly more images, the model may become biased.
Image Quality & Variability: Variations in lighting, angle, or resolution can impact model performance.

Installation & Setup

Clone the repository:

git clone https://github.com/your-repo/welding-defect-detection.git
cd welding-defect-detection

Install dependencies:

pip install -r requirements.txt
Set up MongoDB and configure the database connection.
Run the Flask application:
python app.py

Usage
Open the web application in a browser.
Login or sign up.
Upload an image to analyze for welding defects.
View the prediction results.

Future Improvements

Expand the dataset to improve model generalization.
Experiment with convolutional neural networks (CNNs) for enhanced accuracy.
Implement data augmentation techniques to mitigate overfitting.
Optimize the Flask application for better performance.

