# 🟢 Object Detection AI System

Welcome to the **Object Detection AI System**!  
This project lets you train a custom AI model to recognize four everyday objects using your webcam and deep learning.

---

## 🚀 System Overview

This system uses a Convolutional Neural Network (CNN) built with **Keras** and **TensorFlow** to detect:

- 🥤 **WATERBOTTLE**
- 📱 **PHONE**
- 👓 **GLASSES**
- 🖱️ **MOUSE**

You can collect your own images, train the model, and run real-time detection—all from your computer!

---

## 🛠️ Features

- **Easy Data Collection:** Capture images for each object class using your webcam.
- **Custom Training:** Train a CNN on your own dataset.
- **Live Detection:** See real-time predictions with confidence scores.
- **Visual Feedback:** View sample images and training progress.

---

## 📦 Requirements

- Python 3.x
- [matplotlib](https://matplotlib.org/)
- [numpy](https://numpy.org/)
- [opencv-python](https://pypi.org/project/opencv-python/)
- [tensorflow](https://www.tensorflow.org/)
- [keras](https://keras.io/)
- [seaborn](https://seaborn.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)
- [keras_preprocessing](https://pypi.org/project/Keras-Preprocessing/)

**Install all dependencies:**
```sh
pip install matplotlib numpy opencv-python tensorflow keras seaborn scikit-learn keras_preprocessing
```

---

## 📂 Folder Structure

```
Object-Detection/
│
├── img_1/   # WATERBOTTLE images
├── img_2/   # PHONE images
├── img_3/   # GLASSES images
├── img_4/   # MOUSE images
├── AI Object Detection.py
├── AI_load.py
├── my_model.h5
└── README.md
```

---

## 🏁 How to Use

### 1. **Prepare Folders**

Create four folders in your project directory:
- `img_1` (WATERBOTTLE)
- `img_2` (PHONE)
- `img_3` (GLASSES)
- `img_4` (MOUSE)

### 2. **Collect Training Images**

Run:
```sh
python "AI Object Detection.py"
```
- The webcam will open.
- Press `1`, `2`, `3`, or `4` to save an image to the corresponding folder.
- Press `q` to finish collecting images.

### 3. **Train the Model**

After quitting, the script will:
- Crop, resize, and save your images.
- Train the CNN model.
- Save the model as `my_model.h5`.

### 4. **Run Real-Time Detection**

Run:
```sh
python AI_load.py
```
- The webcam will open.
- The system will display predictions and confidence for the object in view.
- Press `q` to quit.

---

## 💡 Tips

- Collect at least 20–30 images per class for better results.
- Use different backgrounds and lighting for more robust training.
- You can adjust image size and model parameters in the scripts.

---

## 👤 Author

**John Paul Sapasap**

