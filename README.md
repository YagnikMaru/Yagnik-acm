# CP Difficulty & Score Predictor 🎯

**Author:** Yagnik Maru  
**Project Type:** Machine Learning + Full Stack (Flask API)  
**Version:** 5.0.0  

---

## 📌 Project Overview

This project focuses on **automatically predicting the difficulty level and score of competitive programming problems** using Machine Learning.

The system takes a programming problem’s **title, description, input, and output** as input and predicts:
- **Problem Difficulty:** Easy / Medium / Hard  
- **Problem Score:** A continuous numerical value within a realistic range  

The project is designed to be **accurate, stable, and production-ready**, making it suitable for competitive programming platforms and educational tools.

---

## 📂 Dataset Used

- **Format:** JSON Lines (`.jsonl`)
- **Fields Used:**
  - `title`
  - `description`
  - `input`
  - `output`
  - `problem_class` (Easy / Medium / Hard)
  - `problem_score` (Numerical score)

The dataset contains a diverse set of programming problems with varying constraints, algorithms, and difficulty levels.

---

## 🧠 Approach & Models Used

### 🔹 Overall Architecture (Two-Stage Pipeline)

1. **Stage 1 – Classification**
   - Predicts difficulty class: **Easy / Medium / Hard**
   - Model Used: `RandomForestClassifier`
   - Probability calibration applied using `CalibratedClassifierCV`

2. **Stage 2 – Global Score Regression**
   - A **single global regressor** trained on all problems
   - Difficulty class is used as an **input feature**
   - Models evaluated:
     - Gradient Boosting Regressor
     - Random Forest Regressor
     - XGBoost Regressor (if available)
   - Best model selected using **variance-aware composite scoring**

3. **Soft Constraints**
   - Final score is softly constrained based on predicted difficulty
   - Avoids hard clipping while maintaining realistic score ranges

---

## 🛠 Feature Engineering

Along with TF-IDF and Count Vectorizer features, the model uses:
- Text length and word statistics  
- Constraint magnitude (log-scaled)  
- Algorithmic keywords  
- Data structure mentions  
- Optimization and edge-case indicators  
- Mathematical and complexity-related terms  
- Class-based interaction features  

This allows **meaningful score variation within the same difficulty level**.

---

## 📊 Evaluation Metrics

### Classification Metrics
- **Accuracy**
- **Precision, Recall, F1-score** (per class)

### Regression Metrics
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **R² Score**
- **Variance Ratio (Predicted vs True scores)**

These metrics ensure both **accuracy and diversity** in predictions.

---

## 🌐 Web Interface & API

The project includes a **Flask-based production API** with the following features:

### Available Endpoints
- `GET /` – Web Interface / Service Info  
- `POST /predict` – Single problem prediction  
- `POST /batch` – Batch predictions  
- `GET /health` – Health check  
- `GET /info` – Model and feature information  

### Web Interface
- Simple form-based UI
- Accepts problem details
- Displays predicted difficulty, score, confidence, and probability distribution

CORS support and input validation are included for robustness.

---

## ▶️ Steps to Run the Project Locally

### 1️⃣ Clone the Repository
git clone <your-repo-link>
cd CP-Difficulty-Predictor

### 2️⃣ Install Dependencies
pip install -r requirements.txt

### 3️⃣ Train the Models
python train.py

### 4️⃣ Start the Flask Server
python app.py


The server will start at:

http://localhost:5000
