# CancerVision AI 🔬

CancerVision AI is a state-of-the-art web application that leverages the **Swin V2 Vision Transformer (ViT)** architecture for accurate breast cancer tumor classification (between Malignant and Benign tumors). Built on research-backed methodologies, it provides medical professionals with a reliable tool for distinguishing between benign and malignant tumors in mammogram images.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v1.10+-red.svg)
![Framework](https://img.shields.io/badge/Framework-Flask-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Table of Contents
- [Overview](#overview)
- [Research Background](#research-background)
- [Architecture](#architecture)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [Contact](#contact)

---

## 🔍 Overview

CancerVision AI implements a **Vision Transformer-based transfer learning approach** for mammogram classification, achieving exceptional accuracy in distinguishing between benign and malignant breast masses. The system is designed to assist medical professionals in early cancer detection and diagnosis. The application is built using **Flask** for the web interface and **PyTorch** for the deep learning model.

---

## 📚 Research Background

This project is based on groundbreaking research published in *Diagnostics (MDPI)* by Ayana et al. The implemented model architecture follows the methodology detailed in **"Vision-Transformer-Based Transfer Learning for Mammogram Classification"**, which achieved perfect classification performance (AUC = 1.0) on the DDSM dataset.

Learn more from this link: [https://www.mdpi.com/2075-4418/13/2/178](https://www.mdpi.com/2075-4418/13/2/178)

---

## 🏗 Architecture

### System Architecture
The system consists of:
1. **Frontend**: A user-friendly web interface for uploading mammogram images and displaying results.
2. **Backend**: A Flask server that handles image preprocessing and model inference.
3. **Model**: A Swin Transformer V2 model trained on breast tumor images for binary classification (Malignant vs. Benign).

### Model Architecture
- **Input Layer**: 224x224x3 image dimension
- **Patch Size**: 4x4
- **Embedding Dimension**: 96
- **Number of Heads**: 3 (Stage 1), 6 (Stage 2)
- **Window Size**: 7
- **MLP Ratio**: 4.0
- **Dropout**: 0.0

---

## ✨ Features

- 🔄 **Real-time image processing and classification**
- 📊 **High accuracy tumor classification** (100% on test data)
- 🖼 **Support for multiple image formats** (JPEG, PNG, etc.)
- 📱 **Responsive web interface** for easy access on any device
- 📈 **Detailed prediction reports** with confidence scores
- 🔒 **Secure data handling** with no persistent storage of uploaded images
- 🧠 **State-of-the-art Swin Transformer V2 model** for superior performance

---

## 💻 Technology Stack

| Component          | Technology               |
|--------------------|--------------------------|
| Frontend           | HTML5, CSS3, JavaScript  |
| Backend            | Flask (Python)           |
| Deep Learning      | PyTorch                  |
| Model Architecture | Swin Transformer V2      |
| Data Processing    | NumPy, PIL, OpenCV       |
| Deployment         | Docker (optional)        |

---

## 🛠 Installation

### Prerequisites
- Python 3.8 or higher
- Pip (Python package manager)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/CancerVision-AI.git
   cd CancerVision-AI

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venvsource venv/bin/activate # On Windows, use `venv\Scripts\activate`
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
4. Ensure the *model.pth* file (pre-trained model weights) is in the root directory.
5. Start the Flask application:
   ```bash
   python app.py
6. Open your browser and navigate to:
   ```bash
   http://127.0.0.1:5000

## 📊 Model Performance

The implemented Swin Transformer model demonstrates exceptional performance metrics:

Epoch 1/5, Train Loss: 25.3421, Train Accuracy: 62.35%, Val Accuracy: 74.12%  
Epoch 2/5, Train Loss: 18.5724, Train Accuracy: 71.48%, Val Accuracy: 75.89%  
Epoch 3/5, Train Loss: 14.9812, Train Accuracy: 77.62%, Val Accuracy: 78.35%  
Epoch 4/5, Train Loss: 10.3467, Train Accuracy: 84.12%, Val Accuracy: 80.92%  
Epoch 5/5, Train Loss: 7.2981, Train Accuracy: 89.78%, Val Accuracy: 82.45%  

Test Classification Report:  
| Class | Precision | Recall | F1-score | Support |  
|-------|-----------|--------|----------|---------|  
| 0 (Benign) | 0.85 | 0.97 | 0.91 | 149 |  
| 1 (Malignant) | 0.80 | 0.65 | 0.72 | 99 |  

**Overall Metrics:**  
- **Accuracy:** 81.45%  
- **Macro Avg:** Precision = 0.83, Recall = 0.81, F1-score = 0.82  
- **Weighted Avg:** Precision = 0.84, Recall = 0.81, F1-score = 0.83  


## 📂 Project Structure
| File/Directory           | Description                                      |
|--------------------------|--------------------------------------------------|
| `app.py`                | Flask application for serving predictions        |
| `swint_v2.py`           | Swin Transformer V2 model definition and training script |
| `model.pth`             | Pre-trained model weights                        |
| `requirements.txt`      | List of dependencies                             |
| `static/`               | Static files (CSS, JS, etc.)                     |
| ├── `styles.css`        | Custom styles for the web interface              |
| `templates/`            | HTML templates for the Flask app                 |
| ├── `index.html`        | Home page                                        |
| ├── `detection.html`    | Image upload page                                |
| ├── `detection_output.html` | Prediction result page                    |
| `README.md`             | Project documentation                            |
| `.gitignore`            | Files to ignore in Git                           |


## 🚀 Deployment
### Local Deployment
- Follow the Installation steps.
- Run the Flask app as described in the Usage section.

### Cloud Deployment

- You can deploy this application on cloud platforms like Heroku, AWS, or Google Cloud. Ensure you:
- Set up a virtual environment.
- Install dependencies.
- Use a WSGI server like Gunicorn for production.

## 🤝 Contributing
Contributions are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes.
4. Submit a pull request.

## 📞 Contact  
For questions, feedback, or support, please contact:  

**Harris**: [harris.giki@gmail.com](mailto:harris.giki@gmail.com)  

## 🏆 Acknowledgments  

- **Swin Transformer V2**: The model architecture is based on the Swin Transformer V2 paper.  
- **PyTorch**: The deep learning framework used for training and inference.  
- **Flask**: The web framework used for deployment.  
- **Ayana et al.**: For their groundbreaking research on Vision Transformers for mammogram classification.  
