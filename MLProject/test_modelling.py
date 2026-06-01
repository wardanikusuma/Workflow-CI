import os
import pandas as pd

def test_dataset_exists():
    """Menguji apakah file data_clean.csv hasil preprocessing ada"""
    data_path = "telco_customer_churn_preprocessing/data_clean.csv"
    assert os.path.exists(data_path) == True, f"Dataset tidak ditemukan di {data_path}"

def test_dataset_columns():
    """Menguji apakah kolom target 'Churn' ada di dalam data"""
    data_path = "telco_customer_churn_preprocessing/data_clean.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        assert "Churn" in df.columns, "Kolom target 'Churn' hilang!"