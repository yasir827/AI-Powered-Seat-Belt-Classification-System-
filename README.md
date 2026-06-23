# 🚗 Seat Belt Classification System using YOLOv8

## Overview

This project presents an AI-powered Seat Belt Classification System developed using YOLOv8 Classification and Computer Vision techniques. The model is trained to classify images containing seat belt usage and demonstrates the application of deep learning in intelligent transportation and driver safety monitoring systems.

The project covers the complete machine learning pipeline, including dataset preparation, model training, validation, inference, and performance evaluation.

---

## 🎯 Objectives

* Build a deep learning-based seat belt classification model.
* Automate seat belt recognition from images.
* Apply computer vision techniques for transportation safety.
* Develop a deployment-ready AI solution.

---

## 🏗️ Project Pipeline

```text
Seat Belt Classification System (YOLOv8)

        │
        ▼
Dataset Collection
(Seat Belt Images)
        │
        ▼
Dataset Download
(KaggleHub)
        │
        ▼
Data Exploration
& Verification
        │
        ▼
Data Splitting
(Train 80% | Validation 20%)
        │
        ▼
Dataset Organization
(Class-wise Folders)
        │
        ▼
Data Configuration
(YOLOv8 Classification Format)
        │
        ▼
Model Selection
(YOLOv8n-CLS)
        │
        ▼
Model Training
(50 Epochs)
        │
        ▼
Model Validation
(Accuracy & Loss Evaluation)
        │
        ▼
Best Model Selection
(best.pt)
        │
        ▼
Image Classification
(Seat Belt Recognition)
        │
        ▼
Performance Evaluation
(Top-1 Accuracy, Loss)
        │
        ▼
Inference on New Images
        │
        ▼
Deployment Ready Model
```

---

## 📂 Dataset

### Dataset Name

**Seat Belt Dataset**

### Dataset Source

Dataset downloaded from Kaggle using KaggleHub:

```python
import kagglehub

path = kagglehub.dataset_download(
    "sachinxshrivastav/seatbelt-dataset"
)
```

### Dataset Description

The dataset contains images related to seat belt usage and is designed for image classification tasks. Images are organized into class-specific folders and prepared for YOLOv8 Classification training.

### Dataset Structure

```text
dataset/
│
├── train/
│   └── seat_belt/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
│
└── val/
    └── seat_belt/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

### Data Preparation

The following preprocessing steps were performed:

* Dataset downloaded using KaggleHub
* Image quality verification
* Random image shuffling
* Dataset split into:

  * 80% Training Data
  * 20% Validation Data
* Organized according to YOLOv8 Classification format

### Class Information

| Class ID | Class Name |
| -------- | ---------- |
| 0        | seat_belt  |

### Dataset Statistics

| Property         | Value                |
| ---------------- | -------------------- |
| Dataset Type     | Image Classification |
| Classes          | 1                    |
| Training Split   | 80%                  |
| Validation Split | 20%                  |
| Image Formats    | JPG, PNG, JPEG       |

---

## 🛠️ Technologies Used

* Python
* YOLOv8 Classification
* Ultralytics
* KaggleHub
* OpenCV
* Matplotlib
* Deep Learning
* Computer Vision

---

## ⚙️ Model Configuration

| Parameter  | Value               |
| ---------- | ------------------- |
| Model      | YOLOv8n-CLS         |
| Epochs     | 50                  |
| Image Size | 224 × 224           |
| Batch Size | 16                  |
| Framework  | Ultralytics YOLOv8  |
| Device     | NVIDIA Tesla T4 GPU |

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/AI-Powered-Seat-Belt-Classification-System-using-YOLOv8.git

cd AI-Powered-Seat-Belt-Classification-System-using-YOLOv8
```

### Install Dependencies

```bash
pip install ultralytics kagglehub opencv-python matplotlib
```

---

## 🏋️ Model Training

```python
from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

model.train(
    data="dataset_path",
    epochs=50,
    imgsz=224,
    batch=16,
    device=0
)
```

---

## 📊 Model Evaluation

The model was evaluated using standard image classification metrics.

### Evaluation Metrics

* Top-1 Accuracy
* Validation Loss
* Prediction Confidence

### Performance Summary

| Metric          | Result         |
| --------------- | -------------- |
| Top-1 Accuracy  | Near Perfect   |
| Validation Loss | Very Low       |
| Model Type      | Classification |
| Training Epochs | 50             |

> Note: Performance may vary depending on dataset composition and train-validation split.

---

## 🔍 Inference

### Load Trained Model

```python
from ultralytics import YOLO

model = YOLO("best.pt")
```

### Predict on New Images

```python
results = model.predict(
    source="test_image.jpg",
    save=True,
    show=False
)
```

---

## 📁 Repository Structure

```text
Seat-Belt-Classification-System/
│
├── dataset/
│
├── notebooks/
│
├── runs/
│   └── classify/
│       └── train/
│           └── weights/
│               └── best.pt
│
├── data.yaml
├── train.py
├── predict.py
├── requirements.txt
├── README.md
│
└── assets/
    └── pipeline.png
```

---

## 🎯 Applications

* Driver Safety Monitoring
* Smart Transportation Systems
* Vehicle Occupant Compliance Monitoring
* Fleet Management
* Traffic Safety Analytics
* AI-Based Monitoring Systems

---

## ✅ Project Results

* Dataset successfully processed and organized.
* YOLOv8 Classification model trained successfully.
* High classification performance achieved.
* Successful inference on unseen images.
* Deployment-ready model generated.

---

## 🔮 Future Improvements

* Add **No Seat Belt** class for binary classification.
* Real-time webcam/video classification.
* Driver behavior monitoring integration.
* Web dashboard deployment using Streamlit.
* Edge AI deployment on embedded devices.
* Cloud-based inference system.




## ⭐ Acknowledgements

* Ultralytics YOLOv8
* KaggleHub
* OpenCV Community
* Deep Learning and Computer Vision Research Community
