import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

print("Loading ML dataset...")
X = np.load("processed_ml/X_ml.npy")
y = np.load("processed_ml/y_ml.npy")

print("Dataset loaded:", X.shape, y.shape)

# ==========================================
# Train/Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train:", X_train.shape, "Test:", X_test.shape)

# ==========================================
# Helper: Evaluate model
# ==========================================

def evaluate(model, name):
    print("\n==========", name, "==========")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("Accuracy:", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall:", round(rec, 4))
    print("F1 Score:", round(f1, 4))
    print("\nConfusion Matrix:\n", cm)

    return acc, prec, rec, f1

# ==========================================
# MODEL 1 — Random Forest
# ==========================================

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_results = evaluate(rf, "Random Forest")

# ==========================================
# MODEL 2 — Logistic Regression
# ==========================================

lr = LogisticRegression(
    max_iter=500,
    multi_class="auto",
    solver="lbfgs",
    n_jobs=-1
)
lr.fit(X_train, y_train)
lr_results = evaluate(lr, "Logistic Regression")

# ==========================================
# MODEL 3 — Support Vector Machine
# ==========================================

svm = SVC(kernel="rbf", C=2, gamma="scale")
svm.fit(X_train, y_train)
svm_results = evaluate(svm, "SVM")

# ==========================================
# MODEL 4 — Decision Tree
# ==========================================

dt = DecisionTreeClassifier(
    max_depth=20,
    random_state=42
)
dt.fit(X_train, y_train)
dt_results = evaluate(dt, "Decision Tree")

# ==========================================
# SAVE RESULTS FOR REPORT
# ==========================================

results = pd.DataFrame({
    "Model": ["Random Forest", "Logistic Regression", "SVM", "Decision Tree"],
    "Accuracy": [rf_results[0], lr_results[0], svm_results[0], dt_results[0]],
    "Precision": [rf_results[1], lr_results[1], svm_results[1], dt_results[1]],
    "Recall": [rf_results[2], lr_results[2], svm_results[2], dt_results[2]],
    "F1 Score": [rf_results[3], lr_results[3], svm_results[3], dt_results[3]],
})

results.to_csv("processed_ml/model_results.csv", index=False)

print("\nModel Train Complete! Results saved to processed_ml/model_results.csv")
print(results)
