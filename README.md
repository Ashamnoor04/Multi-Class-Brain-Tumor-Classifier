# Multi-Class-Brain-Tumor-Classifier

![Python](https://img.shields.io/badge/Python-3.9-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green?style=for-the-badge&logo=flask)
![Machine Learning](https://img.shields.io/badge/AI-Ensemble%20Learning-orange?style=for-the-badge)

## 📋 Overview
This project is an advanced medical diagnostic tool designed to assist radiologists in the **Multi-Class Classification** of brain tumors. Unlike standard binary classifiers, this system distinguishes between **Glioma**, **Meningioma**, **Pituitary**, and **No Tumor** classes with high precision.

The system utilizes an **Ensemble of 5 Machine Learning Algorithms** and employs **Texture Analysis (GLCM & LBP)** to extract explainable biological features from MRI scans. It is deployed as a user-friendly Web Application that automatically generates standardized **PDF Medical Reports** for clinical documentation.

---

## 🚀 Key Features
* **Multi-Class Diagnosis:** Accurate classification into 4 distinct categories.
* **Ensemble Intelligence:** Combines the power of **Random Forest, XGBoost, SVM, KNN, and Naive Bayes** for robust decision-making.
* **Explainable AI:** Uses **GLCM (Texture)** and **LBP (Pattern)** features instead of "black-box" deep learning, making the diagnosis more interpretable.
* **Automated PDF Reporting:** Generates a professional, downloadable medical report with patient details and diagnostic confidence scores.
* **Computer Vision Enhancement:** Displays "Original vs. Enhanced" MRI scans side-by-side using OpenCV preprocessing.

---

## 📊 Dataset Used
The model was trained on the **Brain Tumor MRI Dataset** sourced from Kaggle.
* **Source:** [Kaggle: Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
* **Total Images:** ~7,023
* **Classes:** Glioma, Meningioma, Pituitary, No Tumor

---

## 📸 Project Screenshots

### 1. Diagnostic Web Dashboard
*The user-friendly interface for uploading MRI scans and viewing results.*
<img width="1171" height="922" alt="Screenshot 2025-11-20 082310" src="https://github.com/user-attachments/assets/800dcc21-13b5-4926-84f1-c199b3d7697e" />

### 2. Automated PDF Medical Report
*The system generates a downloadable PDF for patient records.*
<img width="486" height="457" alt="Screenshot 2025-11-19 114524" src="https://github.com/user-attachments/assets/05c5ddb6-d659-4756-a6e3-29ec5379d6e4" />


### 3. Model Performance Analysis
<img width="822" height="545" alt="Screenshot 2025-11-19 113756" src="https://github.com/user-attachments/assets/b79fc75c-873d-4bb5-a6eb-f95471f90d15" />

---

## 🛠️ Tech Stack
* **Language:** Python 3.9
* **Web Framework:** Flask
* **Machine Learning:** Scikit-Learn, XGBoost
* **Image Processing:** OpenCV, Scikit-Image
* **Reporting:** FPDF (for PDF generation)
* **Visualization:** Matplotlib, Seaborn

---

## ⚙️ Installation & Usage

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/Brain-Tumor-Analysis.git](https://github.com/YOUR_USERNAME/Brain-Tumor-Analysis.git)
    cd Brain-Tumor-Analysis
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**
    ```bash
    python app.py
    ```

4.  **Access the Dashboard**
    Open your browser and navigate to: `http://127.0.0.1:5000/`

---

## 📈 Experimental Results
We evaluated 5 different algorithms. **Random Forest** and **XGBoost** achieved the highest accuracy, validating the effectiveness of ensemble learning for texture-based classification.

| Algorithm | Accuracy | Precision | Recall |
| :--- | :--- | :--- | :--- |
| **Random Forest** | **90.68%** | **High** | **High** |
| **XGBoost** | **90.53%** | **High** | **High** |
| **KNN** | 73.24% | Mod | Mod |
| **Naive Bayes** | 61.35% | Low | Low |
| **SVM** | 57.94% | Low | Low |

---
