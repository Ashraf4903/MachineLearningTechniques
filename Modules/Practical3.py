# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Step 1: Load dataset
# -------------------------------
df = pd.read_csv("titanic.csv")

# -------------------------------
# Step 2: Data Preprocessing
# -------------------------------

# Handle missing values
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop unnecessary columns
df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True, errors='ignore')

# Encode categorical variables
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# -------------------------------
# Step 3: Feature Selection
# -------------------------------
X = df.drop('Survived', axis=1)
y = df['Survived']

# -------------------------------
# Step 4: Feature Scaling
# -------------------------------
scaler = StandardScaler()
X[['Age', 'Fare']] = scaler.fit_transform(X[['Age', 'Fare']])

# -------------------------------
# Step 5: Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -------------------------------
# Step 6: Initialize Model
# -------------------------------
model = LogisticRegression(max_iter=5000)

# -------------------------------
# Step 7: Train Model
# -------------------------------
model.fit(X_train, y_train)

# -------------------------------
# Step 8: Predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# Step 9: Evaluation
# -------------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))