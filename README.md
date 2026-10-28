# 🧠 Brain Tumor and Alzheimer's Detection

An AI/ML-based medical image analysis project focused on detecting brain-related conditions using image processing and deep learning techniques.

## 📌 Project Overview

Brain Tumor and Alzheimer's Detection explores the use of machine learning and deep learning for analyzing medical images. The repository includes Python scripts, image datasets, and a trained model for image-based prediction.

The brain tumor dataset contains image categories such as:
- Glioma
- Meningioma
- No Tumor
- Pituitary Tumor

The project aims to demonstrate how computer vision and deep learning can support medical image classification and research.

## ✨ Features

- Medical image processing and analysis
- Image classification using deep learning
- Trained model saved in HDF5 format (`my_model.h5`)
- Brain tumor image categories
- Python-based implementation
- Dataset organized into folders
- Prediction and visualization scripts

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Image Processing:** OpenCV
- **Numerical Computing:** NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Version Control:** Git and GitHub

## 📂 Project Structure

```text
Brain-Tumor-and-Alzheimer-s-Detection/
│
├── Data/
├── Dataset_/
├── BainTumour.py
├── Login.py
├── MainFile.py
├── Pred.py
├── Prediction.py
├── my_model.h5
├── dbs.db
├── Ori.png
├── val.png
├── README.md
└── .gitattributes
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/RiddheshS/Brain-Tumor-and-Alzheimer-s-Detection.git
cd Brain-Tumor-and-Alzheimer-s-Detection
```

Install Git LFS if required to download the trained model:

```bash
git lfs install
git lfs pull
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install numpy matplotlib opencv-python scikit-learn seaborn scikit-image pillow tensorflow
```

### 4. Run the project

Review the Python scripts to identify the appropriate application entry point and then run the relevant script. For example:

```bash
python MainFile.py
```

**Note:** Additional dependencies or configuration may be required depending on the selected script and your Python environment.

## 🧪 Model and Dataset

The repository contains image datasets and a trained model named `my_model.h5`. The model and scripts are intended for image classification and experimentation.

Model performance should be evaluated using appropriate validation and test datasets before drawing conclusions about its accuracy.

## 🎯 Learning Outcomes

- Understanding medical image classification
- Exploring deep learning and computer vision
- Working with image datasets
- Using Python machine learning libraries
- Understanding model prediction and visualization
- Managing project files with Git and GitHub

## ⚠️ Disclaimer

This project is intended for educational and research purposes only. It is not a clinically validated diagnostic system and must not be used to diagnose, treat, or make medical decisions about any person.

## 👨‍💻 Author

**RiddheshS**

GitHub: [Brain Tumor and Alzheimer's Detection](https://github.com/RiddheshS/Brain-Tumor-and-Alzheimer-s-Detection)

---

⭐ If you find this project useful, consider starring the repository!
