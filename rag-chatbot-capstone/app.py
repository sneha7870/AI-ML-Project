"""
app.py
--------
Flask backend for the RAG chatbot: serves the chat UI and a /chat API endpoint
that runs the full retrieve-then-generate pipeline.
"""

import os
import traceback
from flask import Flask, request, jsonify, render_template

from rag_engine import answer_question, load_index

app = Flask(__name__)

# Fail fast with a clear error if the index hasn't been built yet.
try:
    load_index()
    INDEX_READY = True
except FileNotFoundError as e:
    INDEX_READY = False
    print(f"WARNING: {e}")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "index_ready": INDEX_READY}), 200


@app.route("/")
def home():
    return render_template("chat.html")


@app.route("/chat", methods=["POST"])
def chat():
    if not INDEX_READY:
        return jsonify({
            "error": "Vector index not found. Run `python ingest.py` first, then restart the app."
        }), 503

    data = request.get_json(silent=True)
    if not data or "message" not in data or not data["message"].strip():
        return jsonify({"error": "Request body must be JSON: {'message': '...'}"}), 400

    query = data["message"].strip()

    try:
        result = answer_question(query)
        return jsonify({
            "answer": result["answer"],
            "sources": result["sources"],
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Something went wrong generating a response: {e}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
