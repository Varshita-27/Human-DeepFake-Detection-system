# 🔍 DeepFake Image Detection (Image-Based)

This project detects whether an uploaded face image is **Real** or **Fake** using a deep learning model trained with MobileNetV2.  
The system works **only for images** and includes a simple **Streamlit web interface** for easy testing.

---

## 🚀 Features
- Image-only deepfake detection  
- Fast MobileNetV2-based model  
- Clean Streamlit UI  
- Confidence score for each prediction  

---

## 🗂️ Folder Structure
```
Code/
│── app.py
│── train.py
│── predict.py
│── deepfake_detection_model.h5
│── venv/
Dataset/
│── Dataset/
    ├── Train/
    ├── Validation/
    └── Test/
```

---

## 🔧 Installation
```bash
pip install -r requirements.txt
```

---

## 🏋️ Train the Model
```bash
python train.py
```

---

## 🔍 Run the App
```bash
streamlit run app.py
```

---

## 🧪 Single Image Prediction
```bash
python predict.py
```

---
## ⚙️ How This Project Works

1. **Training Phase**
   - Images from the dataset are given as **Real** and **Fake**.
   - Each image is resized to **96×96** and normalized.
   - MobileNetV2 extracts features from the image.
   - A small classifier predicts whether the face is Real or Fake.
   - The trained model is saved as `deepfake_detection_model.h5`.

2. **Prediction Phase**
   - When the user uploads an image, it is resized to 96×96.
   - The trained model reads the features and gives a probability score.
   - If the score is above 0.50 → **Real**
   - If below 0.50 → **Fake**

3. **Streamlit Interface**
   - User uploads an image.
   - The app runs the prediction function.
   - Output is shown with:
     - Label (Real / Fake)
     - Confidence score
     - Color highlight (Green = Real, Red = Fake)

## 📌 Note
You can download the Real/Fake face datasets from any available open-source sources such as Kaggle or public research datasets.

