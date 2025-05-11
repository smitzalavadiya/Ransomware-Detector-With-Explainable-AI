# 🛡️ AI-Driven Ransomware Detection System with Explainable AI

A scalable and explainable machine learning solution for detecting ransomware using structured features and real-time prediction APIs.

## 🔍 Features
- 52 structured behavioral features
- Random Forest with GridSearchCV tuning
- SMOTE for data balancing
- 99.85% Accuracy, 1.00 Precision/Recall
- Explainability via LIME
- Web deployment using Flask & Streamlit

## 🚀 Project Structure
- `smit_project.ipynb`: Training and evaluation
- `app/`: Flask API
- `dashboard/`: Streamlit app
- `models/`: Trained model
- `ransomware_data_sample.csv`: Sample dataset

## 🛠️ Setup
```bash
pip install -r requirements.txt
python app/app.py
streamlit run dashboard/dashboard.py
