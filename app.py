from flask import Flask, request, render_template, send_file
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from io import BytesIO
from PIL import Image
import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from tensorflow.keras import layers
import keras
from tensorflow.keras import ops

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['OUTPUT_FOLDER'] = 'static/output'
app.secret_key = 'your_secret_key'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']



# Confidence threshold for detecting unknown tumors
CONFIDENCE_THRESHOLD = 0.50

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')

def preprocess_image(img):
    """
    Preprocess the input image to match the model's requirements
    """
    # Resize to match model's input shape
    img = img.resize((128, 128))  # Match your model's input_shape
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded!', 400

    file = request.files['file']
    
    try:
        # Read and preprocess the image
        img = Image.open(BytesIO(file.read()))
        img_array = preprocess_image(img)
        
        # Make prediction
        predictions = model.predict(img_array)
        
        # Since it's binary classification, we can interpret the sigmoid output
        probability = float(predictions[0][0])
        
        # Determine class based on probability threshold
        if probability >= 0.5:
            predicted_class = "Malignant"
            confidence = probability * 100
        else:
            predicted_class = "Benign"
            confidence = (1 - probability) * 100
            
        result = f'Prediction: {predicted_class} (Confidence: {confidence:.2f}%)'
        return render_template('detection_output.html', result=result)
        
    except Exception as e:
        return f'Error processing image: {str(e)}', 500

if __name__ == '__main__':
    # Ensure upload folder exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    app.run(debug=True)