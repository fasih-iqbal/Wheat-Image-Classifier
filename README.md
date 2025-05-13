# 🌾 Wheat Spike Classification Project

## 📌 Project Overview

The Wheat Spike Classification Project utilizes **machine learning** and **computer vision** techniques to classify wheat varieties from images of wheat spikes. The system extracts key morphological and visual features and leverages a **Random Forest** classifier to categorize wheat into four major types: `Akbar`, `Faisalabad`, `Galaxy`, and `Saher`.

This project aims to support research, breeding programs, and large-scale farming by providing a scalable and automated solution for identifying wheat varieties.

---

## 📁 Dataset Structure

The dataset is organized into four labeled folders, each representing a different wheat variety:

```
Dataset/
│
├── akbar/
├── faislabad/
├── galaxy/
└── Saher/
```

Each folder contains \~700 plus labeled images of wheat spikes (e.g., `akbar-01 (1).jpg`), serving as input for feature extraction.

---

## 🧪 Features Extracted

The system extracts **eight key features** per image using image processing techniques:

1. **Spike Length** – Total length of the spike.
2. **Number of Grains** – Count of grains using watershed segmentation.
3. **Awn Length** – Length of bristle-like structures (awns).
4. **Awn Density** – Density of awns via edge detection.
5. **Color Intensity (Mean Hue)** – Average hue from HSV space.
6. **Grain Size Mean** – Average area of detected grains.
7. **Texture Contrast** – Using GLCM (Gray Level Co-occurrence Matrix).
8. **Grain Density** – Number of grains per unit spike length.

Extracted features are saved in a `features.csv` file for model training.

---

## 🧰 Technical Implementation

### 🔧 Preprocessing

* Images are normalized and resized.
* Morphological and color-based features are extracted using OpenCV and scikit-image.

### 🤖 Model Development

* A **Random Forest classifier** is trained using `features.csv`.
* Performance evaluated using **accuracy**, **F1 score**, and a **confusion matrix**.
* Feature importance is used to improve model accuracy.

### 🖥️ Frontend Interface

A user-friendly GUI allows:

* Uploading wheat spike images.
* Visualizing extracted features.
* Displaying model predictions.
* Exploring classification results interactively.

---

## 🚀 Getting Started

### ✅ Prerequisites

Ensure Python 3.9+ is installed. Then, install required libraries:

```bash
pip install opencv-python scikit-learn scikit-image numpy pandas seaborn matplotlib
```

### 📦 Project Structure

```
wheat-spike-classification/
│
├── Dataset/                 # Image folders
├── APP ├── model.ipynb      # Extracts 8 features into features.csv and then trains the model
        ├── features.csv     # Auto-generated CSV of image features
        ├── server.py        # backend 
        ├── app.hml          # front end
        ├── wheat_classifier_model.joblib # model api
└── README.md                # Project documentation
```

### 🛠️ Usage

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd wheat-spike-classification
   ```

2. **Place the dataset**:
   Ensure the `Dataset/` folder with subfolders is in the root directory.

3. **Extract features**:

   ```bash
   python model.ipynb
   ```

4. **Train and evaluate model**:

   ```bash
   python  python model.ipynb
   ```

---

## 📊 Results

* **Initial Accuracy**: \~72% with fewer features.
* **Enhanced Accuracy**: Improved by ~99% after selecting 8 key features.
* **Confusion Matrix**: Provides detailed insight into misclassifications across wheat types.

---

## 🔮 Future Improvements

* 🔍 **Feature Selection**: Use advanced techniques for refining input features.
* 🧠 **Model Tuning**: Try alternative classifiers (SVM, Gradient Boosting) and optimize hyperparameters.
* 🖼️ **Advanced Preprocessing**: Handle varying lighting, noise, and shadows in images.
* 🌾 **Expand Dataset**: Include more varieties and images under different conditions.

---

