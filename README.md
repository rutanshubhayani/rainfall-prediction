# 🌧️ Rainfall Prediction using Machine Learning

This project predicts whether it will rain today based on historical weather conditions.

## 📂 Dataset
- Source: [GeeksforGeeks Rainfall Dataset](https://media.geeksforgeeks.org/wp-content/uploads/20240510131249/Rainfall.csv)  
- Automatically downloaded if not found locally.

## 🛠️ Models Used
- Logistic Regression  
- Support Vector Machine (RBF kernel)  
- XGBoost  

## 📊 Results
| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|----------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 86.5%    | 91.7%     | 88.0%  | **89.8%** | **0.91** |
| SVM (RBF)           | 85.1%    | 85.5%     | **94.0%** | 89.5% | 0.50 |
| XGBoost             | 75.7%    | 79.6%     | 86.0%  | 82.7% | 0.83 |

**Best Model:** Logistic Regression (saved as `rain_model.pkl`)

## 🚀 How to Run
```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn joblib
python rainfall_gfg.py

### Recommended setup (virtual environment)

```bash
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows PowerShell

pip install --upgrade pip
pip install -r requirements.txt
python rainfall_gfg.py
