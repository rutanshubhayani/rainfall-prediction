# 🌧️ Rainfall Prediction using Machine Learning (Python)

This project implements **Rainfall Prediction** using multiple machine learning models, based on the tutorial from [GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/rainfall-prediction-using-machine-learning-python/).

It demonstrates:
- Data preprocessing and feature encoding  
- Handling class imbalance with **RandomOverSampler**  
- Training and evaluating models:  
  - Logistic Regression  
  - Support Vector Machine (SVM)  
  - XGBoost  
- Model comparison using Accuracy, Precision, Recall, F1-score, and ROC-AUC  
- Saving the best model for later use  

---

## 📂 Project Structure

├── rainfall_gfg.py # Main Python script
├── Rainfall.csv # Dataset (optional, auto-downloaded if missing)
├── venv/ # Virtual environment (not uploaded to GitHub)
├── requirements.txt # Dependencies
└── README.md # Project documentation


---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/rainfall-prediction.git
cd rainfall-prediction
```

2. Create Virtual Environment (recommended)
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows PowerShell

3. Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt

4. Run the Script
python rainfall_gfg.py


📊 Output

The script trains and evaluates ML models.
A leaderboard is printed comparing models by F1-score.
The best model is saved as:
rain_model.pkl


📑 Dataset

The script uses Rainfall.csv.
If not present, it automatically downloads the dataset from GeeksforGeeks.
You can also manually download and place it in the project folder.


📝 Requirements

Create a requirements.txt file with the following:
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
joblib


Then install with:
pip install -r requirements.txt


🚀 Future Enhancements

Add more EDA (Exploratory Data Analysis) with visualizations
Hyperparameter tuning with GridSearchCV/Optuna
Deploy model via Flask/Django/FastAPI
Create a Streamlit web app for interactive predictions


🙌 Acknowledgements

GeeksforGeeks Article
Libraries: pandas, numpy, scikit-learn, xgboost, imbalanced-learn