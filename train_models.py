import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)

# Load dataset
df = pd.read_csv("dataset/cleaned_creditcard.csv")

# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Balance training sample weights
weights = compute_sample_weight("balanced", y_train)

# Define algorithms
models = {
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    )
}

results = []
os.makedirs("models", exist_ok=True)

best_score = -1
best_model = None
best_name = ""

# Train and evaluate
for name, model in models.items():

    print("\nTraining:", name)

    if name == "Gradient Boosting":
        model.fit(X_train, y_train, sample_weight=weights)
    else:
        model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    pr_auc = average_precision_score(y_test, probabilities)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("PR-AUC:", pr_auc)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    results.append({
        "Algorithm": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "PR-AUC": pr_auc
    })

    if pr_auc > best_score:
        best_score = pr_auc
        best_model = model
        best_name = name

# Save comparison
comparison = pd.DataFrame(results)
comparison.to_csv("models/model_comparison.csv", index=False)

# Save best model
joblib.dump(best_model, "models/best_model.pkl")

print("\nBest Algorithm:", best_name)
print("Best PR-AUC:", best_score)
print("\nModel Training Completed Successfully!")