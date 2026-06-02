import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import dagshub

print("Memulai eksperimen pelacakan model dengan MLflow & DagsHub...")

# 1. Inisialisasi Koneksi ke DagsHub
REPO_OWNER = "wardanikusuma"
REPO_NAME = "Eksperimen_SML_Tia-Kusuma-Wardani"

#dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
#mlflow.set_tracking_uri(f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow")

# 2. Set Nama Eksperimen di MLflow
mlflow.set_experiment("Telco_Customer_Churn_Modelling")


mlflow.autolog()

# 3. Memuat Data Bersih Hasil Preprocessing Kriteria 1
data_path = "telco_customer_customer_preprocessing/data_clean.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset bersih tidak ditemukan di {data_path}. Pastikan Kriteria 1 sudah dijalankan!")

df = pd.read_csv(data_path)

# 4. Memisahkan Fitur dan Target
X = df.drop(columns=['Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# =====================================================================
# 5. Proses Training Model (Sesuai Standar MLflow Project & Autolog)
# =====================================================================
n_estimators = 100
max_depth = 10
random_state = 42

# Inisialisasi model dan training langsung
# (mlflow.autolog() di atas otomatis merekam parameter, metrik, dan model ke DagsHub)
model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
model.fit(X_train, y_train)

# Evaluasi Model
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n🚀 Eksperimen Berhasil Dicatat dengan Autolog!")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-Score : {f1:.4f}")
print("✨ Silakan cek dashboard DagsHub kamu untuk melihat hasilnya secara online!")