from flask import render_template, request, jsonify
from app import app
from app.utils.mask_detector import detector
from werkzeug.utils import secure_filename
import os
import base64
import io
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import cv2  # Added OpenCV for face detection

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict_webcam', methods=['POST'])
def predict_webcam():
    if not request.json or 'image' not in request.json:
        return jsonify({'error': 'No image data', 'success': False})
    
    try:
        # Extract base64 image
        image_data = request.json['image'].split(',')[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))

        # Preprocess for your model
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        tensor = transform(image).unsqueeze(0).to(detector.device)
        
        with torch.no_grad():
            outputs = detector.model(tensor)
            _, pred = torch.max(outputs, 1)
            label = detector.labels[int(pred)]
        
        # Face detection logic
        face_box = detect_face(image)
        
        return jsonify({
            'label': label,
            'face_box': face_box,  # Face box info will be sent here
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False})

# Function for face detection using OpenCV
def detect_face(image):
    # Convert PIL image to OpenCV format
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return {
            'x': x / cv_image.shape[1],  # Normalize coordinates
            'y': y / cv_image.shape[0],
            'width': w / cv_image.shape[1],
            'height': h / cv_image.shape[0]
        }
    return None

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', error='No selected file')

    if not allowed_file(file.filename):
        return render_template('index.html', error='Allowed file types are png, jpg, jpeg')

    try:
        result = detector.predict(file)
        if result['success']:
            return render_template('index.html', prediction=result['label'])
        else:
            return render_template('index.html', error=result['error'])
    except Exception as e:
        return render_template('index.html', error=str(e))
