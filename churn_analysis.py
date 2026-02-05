
---



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("customer_churn_data.csv")

print("Dataset Preview:")
print(df.head())

# -----------------------------
# Data Cleaning
# -----------------------------
df.dropna(inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# -----------------------------
# Exploratory Data Analysis
# -----------------------------
print("\nChurn Distribution:")
print(df["Churn"].value_counts())

# Churn count plot
plt.figure()
sns.countplot(x="Churn", data=df)
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Tenure vs Churn
plt.figure()
sns.boxplot(x="Churn", y="Tenure", data=df)
plt.title("Tenure vs Churn")
plt.tight_layout()
plt.show()

# Monthly Charges vs Churn
plt.figure()
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges vs Churn")
plt.tight_layout()
plt.show()

# -----------------------------
# Model Building
# -----------------------------
X = df[["Tenure", "MonthlyCharges", "TotalCharges"]]
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
