"""
app.py
--------
Flask app that serves a trained ML model two ways:
  1. A simple HTML form for humans (GET/POST /)
  2. A JSON REST API for programmatic use (POST /predict)
Plus a /health endpoint, which Render (and any host) uses for health checks.
"""

import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

MODEL_PATH = "model.pkl"
META_PATH = "model_meta.pkl"

model = joblib.load(MODEL_PATH)
meta = joblib.load(META_PATH)
FEATURE_NAMES = meta["feature_names"]
CLASS_NAMES = meta["class_names"]


@app.route("/health")
def health():
    """Health check endpoint — Render pings this to confirm the service is alive."""
    return jsonify({"status": "ok"}), 200


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            values = [float(request.form[name]) for name in FEATURE_NAMES]
            pred_idx = model.predict([values])[0]
            probs = model.predict_proba([values])[0]
            prediction = {
                "class": CLASS_NAMES[pred_idx],
                "confidence": round(float(probs[pred_idx]) * 100, 2),
            }
        except (ValueError, KeyError) as e:
            error = f"Invalid input: {e}"

    return render_template(
        "index.html", feature_names=FEATURE_NAMES, prediction=prediction, error=error
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    JSON API endpoint.
    Expects: {"features": [f1, f2, f3, f4]}  (order matching FEATURE_NAMES)
    Returns: {"prediction": "setosa", "confidence": 97.5, "probabilities": {...}}
    """
    data = request.get_json(silent=True)
    if not data or "features" not in data:
        return jsonify({"error": "Request body must be JSON: {'features': [f1, f2, ...]}"}), 400

    features = data["features"]
    if len(features) != len(FEATURE_NAMES):
        return jsonify({
            "error": f"Expected {len(FEATURE_NAMES)} features ({FEATURE_NAMES}), got {len(features)}"
        }), 400

    try:
        features = [float(f) for f in features]
    except (ValueError, TypeError):
        return jsonify({"error": "All features must be numeric"}), 400

    pred_idx = model.predict([features])[0]
    probs = model.predict_proba([features])[0]

    return jsonify({
        "prediction": CLASS_NAMES[pred_idx],
        "confidence": round(float(probs[pred_idx]) * 100, 2),
        "probabilities": {CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(probs)},
    })


if __name__ == "__main__":
    # Local dev only — Render uses gunicorn via the Procfile/start command instead.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
