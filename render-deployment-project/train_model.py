"""
train_model.py
----------------
Trains a small, fast demo model (Iris flower classification) so this template
is 100% self-contained — no dataset download needed. Swap this out for YOUR
trained model (e.g. the Adult Income classifier, Placement Predictor, etc.) —
everything downstream (app.py, Dockerfile, Render config) stays the same as
long as you save a scikit-learn-compatible model with joblib.
"""

import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
class_names = iris.target_names.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, preds):.4f}")
print(classification_report(y_test, preds, target_names=class_names))

joblib.dump(model, "model.pkl")
joblib.dump({"feature_names": feature_names, "class_names": class_names}, "model_meta.pkl")
print("Saved model.pkl and model_meta.pkl")
