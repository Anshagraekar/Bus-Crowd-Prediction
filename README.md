# Bus-Crowd-Prediction
A real-time intelligent system that predicts bus crowd levels by extracting ticket information using OCR and applying machine learning models on historical travel patterns. This project helps passengers avoid overcrowded buses and supports transport authorities in managing public transit more efficiently.
# 🚍 Smart Bus Crowd Prediction System using OCR & Machine Learning

A real-time intelligent system that predicts bus crowd levels by extracting ticket information using Optical Character Recognition (OCR) and applying Machine Learning models on historical travel patterns. This project helps passengers avoid overcrowded buses and supports transport authorities in making data-driven operational decisions.

---

## 🎯 Project Objective

To build a smart public transportation solution that:
- Predicts crowd levels (LOW / HIGH / PEAK)
- Extracts route and time details from printed bus tickets using OCR
- Suggests when and where crowd levels will reduce during a journey
- Supports both manual input and image-based ticket scanning

---

## ⚙️ Key Features

- 📸 OCR-based ticket scanning using Tesseract
- 🧠 Random Forest machine learning model for crowd prediction
- 🕒 Time-based peak and off-peak analysis
- 🚏 Route-aware boarding and alighting insights
- 🌐 Interactive Streamlit web dashboard
- 🧪 Synthetic dataset generation for controlled experiments

---

## 🏗️ System Workflow

1. User inputs route details manually or uploads a ticket image  
2. OCR extracts route ID, boarding stop, destination stop, and time  
3. Extracted data is cleaned and validated  
4. ML model predicts current crowd level  
5. System identifies the stop after which crowd is expected to reduce  
6. Results are displayed on the Streamlit dashboard  

---

## 🛠️ Technology Stack

| Category | Tools |
|---------|-------|
| Programming Language | Python |
| Frontend | Streamlit |
| OCR | Tesseract, OpenCV, Pytesseract |
| Machine Learning | Scikit-learn, Pandas, NumPy |
| Model | Random Forest Classifier |
| Data Storage | CSV, Pickle (.pkl) files |

---

## 📊 Output Example
Route: 61
Boarding Stop: Ujjwal Nagar
Destination: Ashokwan
Time: 09:30 AM

Crowd Level: 🔴 PEAK
Insight: Crowd likely reduces after Ashokwan


---

## 🚀 How to Run the Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/bus-crowd-prediction.git
cd bus-crowd-prediction

pip install -r requirements.txt

streamlit run app.py

http://localhost:8501

bus_crowd_project/
│── app.py
│── predict_utils.py
│── ocr_utils.py
│── train_model.py
│── bus_crowd_with_alighting_14_days.csv
│── crowd_rf_model.pkl
│── label_encoder.pkl
│── requirements.txt
│── README.md
│── LICENSE
│── .gitignore
│── tickets/
│   ├── sample_ticket1.png
│   ├── sample_ticket2.png



