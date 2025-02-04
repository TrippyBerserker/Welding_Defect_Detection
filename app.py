import os
from PIL import Image
from flask import Flask, request, jsonify, redirect, url_for, render_template
import numpy as np
from werkzeug.utils import secure_filename
from pymongo import MongoClient
import bcrypt
import torch
import torchvision.transforms as transforms
from model import ANNModel  # Import ANNModel from model.py

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['auth_demo']
users_collection = db['users']

# Load the model
model = ANNModel()
state_dict = torch.load('model.pth', weights_only=True)  # Load model weights with weights_only=True
model.load_state_dict(state_dict)  # Load state dictionary into the model
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),  # Match model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if user already exists
        if users_collection.find_one({'username': username}):
            return jsonify({'error': 'Username already exists'}), 400

        # Hash the password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Insert user into MongoDB
        users_collection.insert_one({
            'username': username,
            'password': hashed_password
        })

        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Find user in MongoDB
        user = users_collection.find_one({'username': username})

        if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
            return redirect(url_for('user'))  # Redirect to user.html on successful login
        else:
            return jsonify({'error': 'Invalid credentials'}), 401

    return render_template('login.html')

@app.route('/user')
def user():
    return render_template('user.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            print(f"File saved at: {filepath}")  # Debugging Step 1

            # Load and preprocess the image
            image = Image.open(filepath).convert('L')  # Convert to grayscale
            image = transform(np.array(image))  # Apply transformations
            image_tensor = image.unsqueeze(0)  # Add batch dimension

            print("Image Preprocessing Complete")  # Debugging Step 2

            # Perform inference
            with torch.no_grad():
                output = model(image_tensor)  # Forward pass
                print(f"Model Output: {output}")  # Debugging Step 3

                predicted = torch.argmax(output, dim=1).item()  # Get predicted class
                print(f"Predicted Class: {predicted}")  # Debugging Step 4

            # Convert prediction to human-readable result
            result = 'defect' if predicted == 0 else 'non-defect'
            print(f"Final Prediction: {result}")  # Debugging Step 5

        except Exception as e:
            print(f"Error during processing: {e}")  # Debugging Step 6
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
        finally:
            os.remove(filepath)  # Delete the uploaded file after processing

        return jsonify({'prediction': result})
    
    return jsonify({'error': 'Invalid file format. Please upload an image file.'}), 400

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)