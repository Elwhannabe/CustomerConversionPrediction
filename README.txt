This project aims to predict customer conversion using a dataset from a digital marketing campaign. We utilize various data preprocessing techniques, handle class imbalance, and implement machine learning models to achieve accurate predictions.

Prerequisites
Ensure you have the following libraries installed:

numpy
pandas
joblib
matplotlib
seaborn
scikit-learn
imbalanced-learn
You can install these libraries using pip:

bash
Copy code
pip install numpy pandas joblib matplotlib seaborn scikit-learn imbalanced-learn
Project Structure
Data Loading and Initial Inspection
Data Visualization
Data Preprocessing
Model Training and Evaluation
Step-by-Step Guide
1. Data Loading and Initial Inspection
Data Set Upload
python
Copy code
import os
import pandas as pd

file_path = 'D:/BIA_Capstone/CustomerConversionPrediction/data/digital_marketing_campaign_dataset.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist.")
    
data = pd.read_csv(file_path)
Identify Columns
python
Copy code
# Identify categorical and numerical columns
categorical_cols = ['Gender', 'CampaignChannel', 'CampaignType', 'AdvertisingPlatform', 'AdvertisingTool']
numerical_cols = ['Age', 'Income', 'AdSpend', 'ClickThroughRate', 'ConversionRate', 'WebsiteVisits', 'PagesPerVisit', 'TimeOnSite', 'SocialShares', 'EmailOpens', 'EmailClicks', 'PreviousPurchases', 'LoyaltyPoints']
Data Summary
python
Copy code
print(data.describe())
print(data.info())
print(data.isnull().sum())
2. Data Visualization
Visualize the distribution and relationships within the dataset.

Histograms for Numerical Columns
python
Copy code
import matplotlib.pyplot as plt

data[numerical_cols].hist(figsize=(15, 10), bins=20, edgecolor='black')
plt.suptitle('Histograms of Numerical Features')
plt.show()
Box Plots for Numerical Columns
python
Copy code
import seaborn as sns

plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(4, 4, i)
    sns.boxplot(y=data[col])
    plt.title(col)
plt.tight_layout()
plt.show()
Count Plots for Categorical Columns
python
Copy code
plt.figure(figsize=(15, 10))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(3, 2, i)
    sns.countplot(x=col, data=data)
    plt.title(col)
plt.tight_layout()
plt.show()
Correlation Matrix for Numerical Columns
python
Copy code
plt.figure(figsize=(12, 8))
correlation_matrix = data[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.show()
Feature Distributions with Target Variable
python
Copy code
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(4, 4, i)
    sns.histplot(data, x=col, hue='Conversion', kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()
Feature Distributions for Categorical Columns
python
Copy code
plt.figure(figsize=(15, 10))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(3, 2, i)
    sns.countplot(x=col, hue='Conversion', data=data)
    plt.title(col)
plt.tight_layout()
plt.show()
3. Data Preprocessing
Split the Data into Train and Test Sets
python
Copy code
from sklearn.model_selection import train_test_split

X = data.drop('Conversion', axis=1)
y = data['Conversion']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Create Preprocessing Pipelines
python
Copy code
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)
Save Preprocessed Data
python
Copy code
import joblib

model_dir = '../models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
joblib.dump(preprocessor, os.path.join(model_dir, 'preprocessor.joblib'))
Handle Class Imbalance with SMOTE
python
Copy code
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)
4. Model Training and Evaluation
Hyperparameter Tuning with GridSearchCV
python
Copy code
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='f1')
grid_search.fit(X_train_resampled, y_train_resampled)

best_params = grid_search.best_params_
print("Best parameters found: ", best_params)
Train the Random Forest with the Best Parameters
python
Copy code
rf_best_model = RandomForestClassifier(**best_params, random_state=42)
rf_best_model.fit(X_train_resampled, y_train_resampled)
Evaluate the Optimized Random Forest Model
python
Copy code
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

rf_y_pred = rf_best_model.predict(X_test_preprocessed)
rf_y_prob = rf_best_model.predict_proba(X_test_preprocessed)[:, 1]

print("Optimized Random Forest Classifier:")
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_y_pred))
print("\nClassification Report:")
print(classification_report(y_test, rf_y_pred))

rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_y_prob)
rf_roc_auc = auc(rf_fpr, rf_tpr)

plt.figure(figsize=(8, 6))
plt.plot(rf_fpr, rf_tpr, color='darkorange', lw=2, label='Random Forest ROC curve (area = %0.2f)' % rf_roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Random Forest')
plt.legend(loc="lower right")
plt.show()
Train the Gradient Boosting Model
python
Copy code
from sklearn.ensemble import GradientBoostingClassifier

gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train_resampled, y_train_resampled)
Evaluate the Gradient Boosting Model
python
Copy code
gb_y_pred = gb_model.predict(X_test_preprocessed)
gb_y_prob = gb_model.predict_proba(X_test_preprocessed)[:, 1]

print("Gradient Boosting Classifier:")
print("Confusion Matrix:")
print(confusion_matrix(y_test, gb_y_pred))
print("\nClassification Report:")
print(classification_report(y_test, gb_y_pred))

gb_fpr, gb_tpr, _ = roc_curve(y_test, gb_y_prob)
gb_roc_auc = auc(gb_fpr, gb_tpr)

plt.figure(figsize=(8, 6))
plt.plot(gb_fpr, gb_tpr, color='green', lw=2, label='Gradient Boosting ROC curve (area = %0.2f)' % gb_roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Gradient Boosting')
plt.legend(loc="lower right")
plt.show()
Summary
This project involves loading a dataset, performing exploratory data analysis, preprocessing the data, addressing class imbalance, training machine learning models, and evaluating their performance. The visualizations and metrics provide insights into the data and the effectiveness of the models.