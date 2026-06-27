# 📊 Loan-Approval-Prediction-Engine

A supervised machine learning project that predicts whether a loan application will be approved based on applicant information. The project covers the complete ML workflow, including data preprocessing, feature engineering, model training, evaluation, and visualization.

## 🚀 Features

- Data cleaning and preprocessing
- Missing value imputation
- Feature engineering
- Data visualization and exploratory analysis
- Model training and evaluation
- Comparison of multiple classification algorithms
- Hyperparameter tuning
- Random Forest, XGBoost, and SVM models
- Cross-validation

## 📂 Project Structure

```
.
├── SupervisedLearningProject.ipynb
├── loan_approval_data.csv
└── README.md
```

## 🔧 Data Preprocessing

- Removed unnecessary columns (e.g., Applicant ID)
- Dropped records with missing target values
- Mean imputation for numerical features
- Most frequent value imputation for categorical features
- Feature engineering (`Total_Income = Applicant_Income + Coapplicant_Income`)
- Standardization using `StandardScaler`
- Train-Test Split (80:20)

## 📈 Data Visualization

- Feature distribution using histograms
- Exploratory Data Analysis (EDA) for understanding feature distributions and missing values

## 🤖 Models Used

- Logistic Regression
- Decision Tree Classifier

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score

## 🛠️ Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## 📌 Future Improvements

- ROC-AUC analysis
- Model deployment using Flask/FastAPI

---

**Author:** Dip Rajhans Jadhav  
B.Tech Mechanical Engineering, IIT Madras
