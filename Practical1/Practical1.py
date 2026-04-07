# Import required libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Step 1: Load Titanic dataset
df = pd.read_csv("titanic.csv")

print("Original Dataset:\n")
print(df.head())

# -------------------------------
# Step 2: Handle Missing Values
# -------------------------------

# Age → Mean Imputation
imputer = SimpleImputer(strategy='mean')
df['Age'] = imputer.fit_transform(df[['Age']])

# Embarked → Most frequent (mode)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# -------------------------------
# Step 3: Encode Categorical Data
# -------------------------------

# Label Encoding (Sex)
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# One-Hot Encoding (Embarked)
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# -------------------------------
# Step 4: Feature Scaling
# -------------------------------

scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# -------------------------------
# Step 5: Feature Engineering
# -------------------------------

df['Family_Size'] = df['SibSp'] + df['Parch']

# Optional: Drop unnecessary columns
df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True, errors='ignore')

# -------------------------------
# Final Output
# -------------------------------

print("\nPreprocessed Dataset:\n")
print(df.head())