# Pneumonia-Predictor


# Pneumonia Detection Model using DenseNet121

This repository contains a deep learning model built to detect **pneumonia** from chest X-ray images. The model is based on the **DenseNet121** architecture and was fine-tuned to classify X-ray images as either **Pneumonia** or **Normal**. This project aims to assist healthcare professionals by providing an automated tool for early diagnosis.

---

## Table of Contents
- [Overview](#overview)
- [Model Details](#model-details)
- [Installation](#installation)
- [Usage](#usage)
- [Performance](#performance)
- [License](#license)

---

## Overview

This project uses the **DenseNet121** model, a pre-trained deep learning architecture, and fine-tunes it for the task of pneumonia detection. By leveraging transfer learning, the model effectively classifies chest X-ray images into two categories: **Pneumonia** and **Normal**.

---

## Model Details

- **Architecture**: DenseNet121 (Pre-trained on ImageNet and fine-tuned on the pneumonia dataset)
- **Input Data**: Chest X-ray images resized to 224x224 pixels
- **Preprocessing**: Image normalization and augmentation techniques used
- **Framework**: Keras with TensorFlow backend
- **Training**: The model was trained on a **Pneumonia Detection Dataset**, which consists of labeled chest X-ray images.

---

## Installation

To run this project, make sure you have the following installed:

- Python 3.x
- TensorFlow (2.x)
- Keras
- Numpy
- Matplotlib (for visualizations)
- OpenCV

### **Install dependencies**:

Create a virtual environment and install the necessary dependencies using the following:

```
pip install -r requirements.txt
```

---

## Usage

### **Load the Model**:
```python
from tensorflow.keras.models import load_model

model = load_model('final_pneumonia_model.h5')
```

### **Preprocess an Image**:

To use the model for prediction, first, load and preprocess the chest X-ray image:

```python
from tensorflow.keras.preprocessing import image
import numpy as np

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize image
    return img_array
```

### **Make a Prediction**:

Once the image is preprocessed, you can use the model to make a prediction:

## Dataset

The dataset used to train the model consists of chest X-ray images labeled as **Pneumonia** or **Normal**. The images were preprocessed to a consistent size of **224x224 pixels** and normalized to improve training efficiency and model performance.

For privacy reasons, the dataset is not included in this repository, but you can use any suitable pneumonia detection dataset, such as the **Kaggle Chest X-ray Pneumonia dataset**.

---

## Performance

- **Accuracy**: 90%
- **Recall**: 95% (strong ability to detect pneumonia cases)
- **Precision**: Optimized to minimize false positives

This model is well-suited for **early detection** and can assist healthcare professionals in diagnosing pneumonia quickly and accurately.

---

## License

This project is licensed under the MIT License 

