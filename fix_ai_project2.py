# ===============================
# 1. IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("✅ All libraries successfully imported")

# ===============================
# 2. LOAD MAIN DATASET
# ===============================
try:
    df = pd.read_csv("creditcard.csv.csv")   # Change file name if required
    print("✅ Main dataset loaded successfully!")
    print("Shape:", df.shape)

except FileNotFoundError:
    print("❌ Dataset not found! Creating sample dataset...")
    np.random.seed(42)
    sample_data = {
        'Feature1': np.random.normal(0, 1, 1000),
        'Feature2': np.random.normal(0, 1, 1000),
        'Feature3': np.random.normal(0, 1, 1000),
        'Class': np.random.randint(0, 2, 1000)
    }
    df = pd.DataFrame(sample_data)

print(df.head())

# ===============================
# 3. DATA EXPLORATION
# ===============================
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# ===============================
# 4. DATA PREPARATION
# ===============================
df = df.fillna(df.mean(numeric_only=True))

if "Class" not in df.columns:
    df.rename(columns={df.columns[-1]: "Class"}, inplace=True)

X = df.drop("Class", axis=1)
y = df["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"\nTraining size: {X_train.shape}")
print(f"Testing size: {X_test.shape}")

# ===============================
# 5. TRAINING MODEL
# ===============================
model = LogisticRegression(max_iter=5000, random_state=42)
model.fit(X_train, y_train)
print("\n🚀 Model training completed!")

# ===============================
# 6. EVALUATION
# ===============================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n📊 MODEL PERFORMANCE")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# ===============================
# 7. CONFUSION MATRIX
# ===============================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ===============================
# 8. HISTOGRAM & LINE PLOT
# ===============================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(df[df.columns[0]], bins=30, alpha=0.7)
plt.title("Histogram of Feature 1")
plt.xlabel("Value")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.plot(df[df.columns[1]].head(100), marker="o", markersize=2)
plt.title("Line Plot of Feature 2 (first 100 values)")
plt.xlabel("Index")
plt.ylabel("Value")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n📌 Visualizations Completed")

# ===============================
# 9. AUTOMATIC BATCH PREDICTION
# ===============================
print("\n📁 Loading secondary dataset for predictions...")

try:
    new_data = pd.read_csv("creditcard.csv.csv")   # Replace filename if needed
    print("➡️ Prediction dataset loaded successfully!")
    print("Shape:", new_data.shape)
    print(new_data.head())

    # Drop Class column if exists
    if "Class" in new_data.columns:
        new_data = new_data.drop("Class", axis=1)

    # Scale using previously fitted scaler
    new_scaled = scaler.transform(new_data)

    predictions = model.predict(new_scaled)
    probabilities = model.predict_proba(new_scaled)

    results = new_data.copy()
    results["Predicted_Class"] = predictions
    results["Probability_Class_1"] = probabilities[:, 1]
    results["Confidence"] = np.max(probabilities, axis=1)

    results.to_csv("prediction_results.csv", index=False)
    print("\n💾 Predictions saved to: prediction_results.csv")

    print("\n📍 Sample output:")
    print(results.head())

except FileNotFoundError:
    print("❌ 'new_data.csv' not found. Please add the dataset to the same folder.")


print("\n🎉 PROJECT COMPLETED SUCCESSFULLY")
