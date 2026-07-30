"""
Movie Recommendation System
------------------------------
Builds two recommenders on the MovieLens dataset and compares them:
  1. Collaborative filtering (matrix factorization via SVD, using the
     `surprise` library) — recommends based on rating patterns across users.
  2. Content-based filtering (TF-IDF over genres) — recommends based on
     similarity of movie attributes, useful for cold-start (new movies/users).

Dataset: MovieLens 100K (ml-latest-small) — download from
https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
and unzip so `movies.csv` and `ratings.csv` sit in DATA_DIR below.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "ml-latest-small"
MOVIES_CSV = os.path.join(DATA_DIR, "movies.csv")
RATINGS_CSV = os.path.join(DATA_DIR, "ratings.csv")

if not os.path.exists(MOVIES_CSV):
    raise FileNotFoundError(
        f"'{MOVIES_CSV}' not found. Download the MovieLens ml-latest-small dataset from "
        "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip, unzip it, "
        "and place the folder here (or update DATA_DIR)."
    )

movies = pd.read_csv(MOVIES_CSV)
ratings = pd.read_csv(RATINGS_CSV)

print(f"Movies: {movies.shape}, Ratings: {ratings.shape}")
print(f"Unique users: {ratings['userId'].nunique()}, Unique movies rated: {ratings['movieId'].nunique()}")

# ---------------------------------------------------------------------------
# 1. EDA
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(x="rating", data=ratings, ax=axes[0])
axes[0].set_title("Rating Distribution")
ratings_per_user = ratings.groupby("userId").size()
sns.histplot(ratings_per_user, bins=50, ax=axes[1])
axes[1].set_title("Ratings per User")
plt.tight_layout()
plt.savefig("eda_overview.png", dpi=150)
plt.close()

# ===========================================================================
# PART A: COLLABORATIVE FILTERING (Matrix Factorization / SVD)
# ===========================================================================
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split as surprise_split, cross_validate
from surprise import accuracy as surprise_accuracy

reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)

trainset, testset = surprise_split(data, test_size=0.2, random_state=42)

svd_model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
svd_model.fit(trainset)

predictions = svd_model.test(testset)
rmse = surprise_accuracy.rmse(predictions, verbose=True)
mae = surprise_accuracy.mae(predictions, verbose=True)

# 5-fold cross-validation for a more robust estimate
cv_results = cross_validate(svd_model, data, measures=["RMSE", "MAE"], cv=5, verbose=True)
print(f"\nCV RMSE: {np.mean(cv_results['test_rmse']):.4f} (+/- {np.std(cv_results['test_rmse']):.4f})")

movie_id_to_title = dict(zip(movies["movieId"], movies["title"]))

def recommend_collaborative(user_id, n=10):
    """Top-N movies for a user, ranked by predicted rating, excluding already-rated movies."""
    rated_movie_ids = set(ratings[ratings["userId"] == user_id]["movieId"])
    all_movie_ids = set(movies["movieId"])
    candidates = list(all_movie_ids - rated_movie_ids)

    preds = [(mid, svd_model.predict(user_id, mid).est) for mid in candidates]
    preds.sort(key=lambda x: x[1], reverse=True)

    top_n = preds[:n]
    return pd.DataFrame([
        {"movieId": mid, "title": movie_id_to_title.get(mid, "Unknown"), "predicted_rating": round(score, 2)}
        for mid, score in top_n
    ])

print("\n--- Sample: Top 10 collaborative-filtering recommendations for User 1 ---")
print(recommend_collaborative(user_id=1, n=10).to_string(index=False))

# ===========================================================================
# PART B: CONTENT-BASED FILTERING (TF-IDF on genres)
# ===========================================================================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies["genres_clean"] = movies["genres"].str.replace("|", " ", regex=False)
movies.loc[movies["genres_clean"] == "(no genres listed)", "genres_clean"] = ""

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres_clean"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

title_to_idx = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

def recommend_content_based(title, n=10):
    """Top-N movies most similar in genre profile to the given movie title."""
    if title not in title_to_idx:
        matches = movies[movies["title"].str.contains(title, case=False, na=False)]
        if matches.empty:
            return f"No movie found matching '{title}'."
        title = matches.iloc[0]["title"]
        print(f"(Using closest match: '{title}')")

    idx = title_to_idx[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n + 1]
    movie_indices = [i[0] for i in sim_scores]

    return movies.iloc[movie_indices][["movieId", "title", "genres"]].assign(
        similarity=[round(s, 3) for _, s in sim_scores]
    )

print("\n--- Sample: Top 10 content-based recommendations similar to 'Toy Story (1995)' ---")
print(recommend_content_based("Toy Story (1995)", n=10).to_string(index=False))

# ===========================================================================
# PART C: HYBRID (optional simple blend)
# ===========================================================================
def recommend_hybrid(user_id, seed_title, n=10, alpha=0.5):
    """
    Blends collaborative predicted rating with content similarity to a seed movie
    the user liked. alpha weights collaborative (alpha) vs content (1-alpha).
    """
    collab_df = recommend_collaborative(user_id, n=50)
    content_df = recommend_content_based(seed_title, n=50)
    if isinstance(content_df, str):
        return content_df

    merged = collab_df.merge(content_df[["movieId", "similarity"]], on="movieId", how="inner")
    # normalize predicted_rating to 0-1 to combine fairly with similarity
    merged["norm_rating"] = (merged["predicted_rating"] - merged["predicted_rating"].min()) / (
        merged["predicted_rating"].max() - merged["predicted_rating"].min() + 1e-8
    )
    merged["hybrid_score"] = alpha * merged["norm_rating"] + (1 - alpha) * merged["similarity"]
    merged = merged.sort_values("hybrid_score", ascending=False)
    return merged[["title", "predicted_rating", "similarity", "hybrid_score"]].head(n)

print("\n--- Sample: Hybrid recommendations for User 1, seeded on 'Toy Story (1995)' ---")
hybrid_result = recommend_hybrid(user_id=1, seed_title="Toy Story (1995)", n=10)
print(hybrid_result.to_string(index=False) if isinstance(hybrid_result, pd.DataFrame) else hybrid_result)

# ---------------------------------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------------------------------
import pickle
with open("svd_model.pkl", "wb") as f:
    pickle.dump(svd_model, f)
with open("tfidf_artifacts.pkl", "wb") as f:
    pickle.dump({"tfidf": tfidf, "cosine_sim": cosine_sim, "title_to_idx": title_to_idx}, f)

print("\nSaved svd_model.pkl and tfidf_artifacts.pkl")
print("Done.")
