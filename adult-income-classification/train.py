"""
Adult Census Income Classification
------------------------------------
Predicts whether an individual's income exceeds $50K/year based on census data.
Dataset: UCI Adult Census Income Dataset (auto-downloaded).

Pipeline: EDA -> Preprocessing -> Model Training (LogReg, RandomForest, GradientBoosting)
-> Evaluation -> Save best model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
)

RANDOM_STATE = 42
sns.set_style("whitegrid")

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

def load_data():
    """Loads the Adult dataset. Falls back to local CSVs (adult.data / adult.test)
    placed in the same folder if the download fails (e.g. no internet access)."""
    try:
        train_df = pd.read_csv(TRAIN_URL, names=COLUMNS, sep=r",\s*", engine="python", na_values="?")
        test_df = pd.read_csv(TEST_URL, names=COLUMNS, sep=r",\s*", engine="python", na_values="?", skiprows=1)
        test_df["income"] = test_df["income"].str.replace(".", "", regex=False)
        df = pd.concat([train_df, test_df], ignore_index=True)
        print(f"Downloaded dataset from UCI. Shape: {df.shape}")
    except Exception as e:
        print(f"Download failed ({e}). Trying local files adult.data / adult.test ...")
        train_df = pd.read_csv("adult.data", names=COLUMNS, sep=r",\s*", engine="python", na_values="?")
        test_df = pd.read_csv("adult.test", names=COLUMNS, sep=r",\s*", engine="python", na_values="?", skiprows=1)
        test_df["income"] = test_df["income"].str.replace(".", "", regex=False)
        df = pd.concat([train_df, test_df], ignore_index=True)
    return df

df = load_data()

# ---------------------------------------------------------------------------
# 2. EDA
# ---------------------------------------------------------------------------
print("\n--- Basic Info ---")
print(df.info())
print("\n--- Missing values ---")
print(df.isnull().sum()[df.isnull().sum() > 0])
print("\n--- Target distribution ---")
print(df["income"].value_counts(normalize=True))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(x="income", data=df, ax=axes[0])
axes[0].set_title("Income Class Distribution")
sns.histplot(df["age"], bins=30, kde=True, ax=axes[1])
axes[1].set_title("Age Distribution")
plt.tight_layout()
plt.savefig("eda_overview.png", dpi=150)
plt.close()
print("Saved eda_overview.png")

# ---------------------------------------------------------------------------
# 3. PREPROCESSING
# ---------------------------------------------------------------------------
df.dropna(inplace=True)
df.drop(columns=["fnlwgt", "education"], inplace=True)  # education_num already encodes education
df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})

categorical_cols = df.select_dtypes(include="object").columns.tolist()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop(columns=["income"])
y = df["income"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

scaler = StandardScaler()
num_cols = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# ---------------------------------------------------------------------------
# 4. MODEL TRAINING
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE),
}

results = []
fitted_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "ROC-AUC": roc_auc_score(y_test, probs),
    })
    fitted_models[name] = model
    print(f"\n=== {name} ===")
    print(classification_report(y_test, preds, target_names=["<=50K", ">50K"]))

results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
print("\n--- Model Comparison ---")
print(results_df.to_string(index=False))
results_df.to_csv("model_comparison.csv", index=False)

# ---------------------------------------------------------------------------
# 5. BEST MODEL EVALUATION PLOTS
# ---------------------------------------------------------------------------
best_name = results_df.iloc[0]["Model"]
best_model = fitted_models[best_name]
print(f"\nBest model: {best_name}")

preds = best_model.predict(X_test)
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
for name, model in fitted_models.items():
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
plt.title("ROC Curves")
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150)
plt.close()

if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances.values, y=importances.index)
    plt.title(f"Feature Importance - {best_name}")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()

# ---------------------------------------------------------------------------
# 6. SAVE MODEL
# ---------------------------------------------------------------------------
joblib.dump(best_model, "best_income_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
print("\nSaved best_income_model.pkl, scaler.pkl, label_encoders.pkl")
print("Done.")
