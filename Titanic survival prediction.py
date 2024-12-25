# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load the Titanic dataset
df = pd.read_csv('titanic.csv')  # Replace with the actual dataset path

# Data Exploration
print("Dataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# Handle missing data
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Cabin'] = df['Cabin'].fillna('Unknown')

# Feature Engineering
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)  # Extract title from Name
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1  # Family size
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)  # Is passenger alone?
df['FareBin'] = pd.qcut(df['Fare'], 4, labels=[1, 2, 3, 4])  # Bin fares
df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 50, 80], labels=[1, 2, 3, 4])  # Bin ages

# Selecting relevant features
features = ['Pclass', 'Sex', 'AgeBin', 'FareBin', 'Embarked', 'IsAlone']
df = df[features + ['Survived']]

# Encode categorical variables
encoder = LabelEncoder()
for col in ['Sex', 'Embarked', 'AgeBin', 'FareBin']:
    df[col] = encoder.fit_transform(df[col])

# Splitting the data into train and test sets
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training with Random Forest
model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model selection
best_model = grid_search.best_estimator_
print("\nBest Parameters:", grid_search.best_params_)

# Model Evaluation
y_pred = best_model.predict(X_test)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix Visualization
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
