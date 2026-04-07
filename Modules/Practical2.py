# Import required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load dataset
df = pd.read_csv("titanic.csv")

print("Dataset Preview:\n")
print(df.head())

# -------------------------------
# Step 2: Data Preprocessing (Basic)
# -------------------------------

# Handle missing values
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Convert categorical to numeric for correlation
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# -------------------------------
# Step 3: Descriptive Statistics
# -------------------------------

print("\nDescriptive Statistics:\n")
print(df.describe())

# -------------------------------
# Step 4: Correlation Matrix
# -------------------------------

corr = df.corr(numeric_only=True)
print("\nCorrelation Matrix:\n", corr)

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# -------------------------------
# Step 5: Covariance Matrix
# -------------------------------

cov = df.cov(numeric_only=True)
print("\nCovariance Matrix:\n", cov)

# -------------------------------
# Step 6: Visualization
# -------------------------------

# Histogram
df['Age'].hist()
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Boxplot
sns.boxplot(x=df['Fare'])
plt.title("Fare Boxplot")
plt.show()

# Pairplot (selected features for clarity)
sns.pairplot(df[['Age', 'Fare', 'Survived']], hue='Survived')
plt.show()