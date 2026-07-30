"""
ingest.py
-----------
Run this once (and again whenever your documents change) to build the
vector store the chatbot searches at query time.

Usage:
    python ingest.py                  # uses ./data by default
    python ingest.py --data_dir docs  # custom folder
"""

import argparse
from rag_engine import build_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data", help="Folder containing .txt/.md documents")
    args = parser.parse_args()

    build_index(args.data_dir)
    print("\nIndex built. You can now run the chatbot: python app.py")
