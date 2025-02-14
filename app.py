from flask import Flask, request, render_template, send_file
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from io import BytesIO

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['OUTPUT_FOLDER'] = 'static/output'
app.secret_key = 'your_secret_key'

# Function to check allowed file extensions


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Define class labels (ensure 'notumor' is in the last position for easy checking)
class_labels = ['glioma', 'meningioma', 'pituitary', 'notumor']

# Confidence threshold for detecting unknown tumors
CONFIDENCE_THRESHOLD = 0.50


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded!', 400

    file = request.files['file']

    # Read the image file and convert it to a format suitable for keras
    img = image.load_img(BytesIO(file.read()), target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)


    # Check for 'notumor' class or low confidence
    if predicted_class_label == 'notumor':
        result = f'No tumor detected with confidence: {max_probability * 100:.2f}%'
    
    elif max_probability < CONFIDENCE_THRESHOLD:
        result = f'Tumor detected, but type is not in trained classes. Confidence: {max_probability * 100:.2f}%'
    
    else:
        result = f'Tumor detected: {predicted_class_label} with confidence: {max_probability * 100:.2f}%'

    return render_template('detection_output.html', result=result)


@app.route('/upload', methods=['POST'])
def upload():
    if 'image_files[]' not in request.files:
        flash('No file part')
        return redirect(request.url)

    files = request.files.getlist('image_files[]')
    uploaded_files = []

    if not files:
        flash('No selected file')
        return redirect(request.url)

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            uploaded_files.append(file_path)
        else:
            flash('Invalid file type')
            return redirect(request.url)

    if uploaded_files:
        try:
            explanations = []
            for file_path in uploaded_files:
                
                # Add URLs to static output folder
                output_files = {
                    "original": url_for('static', filename='output/original_image.png'),
                    "heatmap": url_for('static', filename='output/heatmap.png'),
                    "superimposed": url_for('static', filename='output/superimposed_image.png'),
                    "lime": url_for('static', filename='output/lime_explanation_image.png')
                }
                explanations.append(output_files)

            return render_template('xai.html', explanations=explanations)
        except Exception as e:
            flash(f"Error processing the image(s): {str(e)}")
            return redirect(request.url)

    flash('No valid images processed')
    return redirect(request.url)



if __name__ == '__main__':
    # Ensure upload folder exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    app.run(debug=True)
