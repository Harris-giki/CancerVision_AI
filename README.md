# CancerVision AI 🔬

CancerVision AI is a state-of-the-art web application that leverages Swin V2 Vision Transformer (ViT) architecture for accurate breast cancer tumor classification (among Maligant and Benign Tumours). Built on research-backed methodologies, it provides medical professionals with a reliable tool for distinguishing between benign and malignant tumors in mammogram images.
![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v1.10+-red.svg)
![Framework](https://img.shields.io/badge/Framework-Flask-green.svg)

## 📋 Table of Contents
- [Overview](#overview)
- [Research Background](#research-background)
- [Architecture](#architecture)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Usage](#usage)
- [Model Performance](#model-performance)


## 🔍 Overview

CancerVision AI implements a Vision Transformer-based transfer learning approach for mammogram classification, achieving exceptional accuracy in distinguishing between benign and malignant breast masses. The system is designed to assist medical professionals in early cancer detection and diagnosis.

## 📚 Research Background

This project is based on groundbreaking research published in Diagnostics (MDPI) by Ayana et al. The implemented model architecture follows the methodology detailed in "Vision-Transformer-Based Transfer Learning for Mammogram Classification" which achieved perfect classification performance (AUC = 1.0) on the DDSM dataset.

Learn more from this link: [https://www.mdpi.com/2075-4418/13/2/178](harris.giki@gmail.com)

## 🏗 Architecture

### System Architecture

### Model Architecture
- **Input Layer**: 128x128x3 image dimension
- **Patch Size**: 4x4
- **Embedding Dimension**: 64
- **Number of Heads**: 8
- **Window Size**: 4
- **MLP Size**: 256

## ✨ Features

- 🔄 Real-time image processing and classification
- 📊 High accuracy tumor classification
- 🖼 Support for multiple image formats
- 📱 Responsive web interface
- 📈 Detailed prediction reports
- 🔒 Secure data handling

## 💻 Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | HTML5, CSS3, JavaScript |
| Backend | Flask (Python) |
| Deep Learning | Pytorch, Keras |
| Model Architecture | Swin Transformer |
| Data Processing | NumPy, OpenCV |
| Deployment | Docker |

## 🎯 Usage

1. Access the web interface at `http://localhost:5000`
2. Upload a mammogram image
3. Click "Analyze" to process the image
4. View the classification results and confidence score

## 📊 Model Performance

The implemented Swin Transformer model demonstrates exceptional performance metrics:

| Metric | Score |
|--------|--------|
| Training Accuracy | 100% |
| Validation Accuracy | 100% |
| Test Accuracy | 100% |
| Processing Time | <2 seconds |

## 📞 Contact

For questions and support, please contact [harris.giki@gmail.com](harris.giki@gmail.com)
