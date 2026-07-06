import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("Titanic.csv")

# Display first five rows
print("========== FIRST 5 ROWS ==========")
print(df.head())

# Display dataset information
print("\n========== DATASET INFO ==========")
df.info()

# Display statistical summary
print("\n========== SUMMARY STATISTICS ==========")
print(df.describe())

# Display missing values before cleaning
print("\n========== MISSING VALUES BEFORE CLEANING ==========")
print(df.isnull().sum())

# Handle missing values
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop Cabin column if it exists
if "Cabin" in df.columns:
    df.drop("Cabin", axis=1, inplace=True)

# Display missing values after cleaning
print("\n========== MISSING VALUES AFTER CLEANING ==========")
print(df.isnull().sum())

# Label Encoding for Sex column
le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])

# One Hot Encoding for Embarked column
df = pd.get_dummies(df, columns=["Embarked"])

# Display cleaned dataset
print("\n========== CLEANED DATASET ==========")
print(df.head())

# Plot Age Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.grid(True)
plt.show()

# Save cleaned dataset
df.to_csv("Titanic_Cleaned.csv", index=False)

print("\nCleaned dataset has been saved as Titanic_Cleaned.csv")