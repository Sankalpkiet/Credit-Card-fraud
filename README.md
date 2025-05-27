# Credit-Card-fraud
# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Step 2: Load Dataset
df = pd.read_csv('creditcard.csv')
print("✅ Data Loaded. Shape:", df.shape)

# Step 3: Data Summary
print("\n🔍 Summary Statistics:")
print(df.describe())

print("\n📊 Class Distribution:")
print(df['Class'].value_counts())

# Step 4: Data Preprocessing
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

# Drop original 'Time' and 'Amount'
df.drop(['Time', 'Amount'], axis=1, inplace=True)

# Rearrange columns
columns = [col for col in df.columns if col != 'Class'] + ['Class']
df = df[columns]

# Step 5: Isolation Forest Model
X = df.drop('Class', axis=1)
y = df['Class']

model = IsolationForest(n_estimators=100, contamination=0.001, random_state=42)
model.fit(X)

# Predict (-1 for fraud, 1 for normal)
pred = model.predict(X)
pred = [1 if x == -1 else 0 for x in pred]  # Convert to 1 = Fraud, 0 = Normal

# Step 6: Evaluation
print("\n✅ Evaluation Metrics:\n")
print(confusion_matrix(y, pred))
print(classification_report(y, pred, target_names=["Genuine", "Fraud"]))

print(f"Accuracy  : {accuracy_score(y, pred) *100:.4f}")
print(f"Precision : {precision_score(y, pred):.4f}")
print(f"Recall    : {recall_score(y, pred):.4f}")
print(f"F1 Score  : {f1_score(y, pred):.4f}")

# Step 7: Confusion Matrix Heatmap
cm = confusion_matrix(y, pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Genuine", "Fraud"], yticklabels=["Genuine", "Fraud"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
