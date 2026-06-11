import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

loan=pd.read_csv('loan_approval_data.csv')
loan=loan.dropna(subset=['Loan_Approved'])
loan=loan.drop('Applicant_ID',axis=1)

## Missing Data

to_mean=['Applicant_Income','Coapplicant_Income','Age','Credit_Score','DTI_Ratio','Savings','Collateral_Value','Loan_Amount','Loan_Term']
to_max=['Employment_Status','Marital_Status','Dependents','Existing_Loans','Loan_Purpose','Property_Area','Education_Level','Gender','Employer_Category']

from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_max=SimpleImputer(strategy='most_frequent')

loan[to_mean]=imp_mean.fit_transform(loan[to_mean])
loan[to_max]=imp_max.fit_transform(loan[to_max])

## Encoding

loan['Gender']=loan['Gender'].map({'Male':1,'Female':0})
loan['Loan_Approved']=loan['Loan_Approved'].map({'Yes':1,'No':0})
loan['Marital_Status']=loan['Marital_Status'].map({'Married':1,'Single':0})
loan['Education_Level']=loan['Education_Level'].map({'Graduate':1,'Not Graduate':0})
loan['Employment_Status']=loan['Employment_Status'].map({'Self-employed':2,'Unemployed':0,'Salaried':3,'Contract':1})
loan['Property_Area']=loan['Property_Area'].map({'Urban':2,'Semiurban':1,'Rural':0})


cat=['Dependents', 'Existing_Loans']
loan[cat]=loan[cat].astype('int64')
loan=pd.get_dummies(loan,columns=['Loan_Purpose','Employer_Category'],dtype='int64')

## Input Out put data seperation
X=loan.drop('Loan_Approved',axis=1)
y=loan['Loan_Approved']
X.info()

## Feature engineering
X['Total_Income']=X['Applicant_Income']+X['Coapplicant_Income']

## Splitting the model in training and testing set 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

## Model Training

from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()

nb.fit(X_train,y_train)
y_pred=nb.predict(X_test)

## Model Testing.

acc_score=accuracy_score(y_test,y_pred)
pre_score=precision_score(y_test,y_pred)
rec_score=recall_score(y_test,y_pred)

print("Accuracy score:",acc_score)
print("Precision score:",pre_score)
print("Recall score:",rec_score)