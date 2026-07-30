# Movie Recommendation System

Builds and compares two recommendation approaches on the MovieLens dataset, plus a simple hybrid blend.

## Dataset
**MovieLens `ml-latest-small`** (100,836 ratings, 9,742 movies, 610 users): download from
https://files.grouplens.org/datasets/movielens/ml-latest-small.zip, unzip, and place the `ml-latest-small` folder next to `train.py` (needs `movies.csv` and `ratings.csv`).

## Approaches

### 1. Collaborative Filtering — SVD (matrix factorization)
Uses the `surprise` library's SVD implementation to learn latent factors for users and movies from the rating matrix, then predicts ratings for unrated movie/user pairs. Recommends movies with the highest predicted rating for a given user. Evaluated with RMSE/MAE and 5-fold cross-validation.

### 2. Content-Based Filtering — TF-IDF on genres
Vectorizes each movie's genre tags with TF-IDF and computes cosine similarity between movies. Recommends movies most similar in genre profile to a movie the user liked. Doesn't need any rating history, so it works for brand-new movies/users (solves collaborative filtering's "cold start" problem).

### 3. Hybrid
Blends the two: takes a user's collaborative-filtering candidates, re-ranks them by how genre-similar they are to a movie the user is known to like, weighted by `alpha`.

## Run
```bash
pip install pandas numpy scikit-learn scikit-surprise matplotlib seaborn
python train.py
```
**Note**: `scikit-surprise` sometimes needs a C++ build toolchain to install (`pip install scikit-surprise` can fail on some systems without it — if so, `conda install -c conda-forge scikit-surprise` is more reliable).

## Outputs
- `eda_overview.png` — rating distribution and ratings-per-user
- Printed: RMSE/MAE, cross-validation scores, and sample recommendations for a test user + movie
- `svd_model.pkl`, `tfidf_artifacts.pkl` — for deployment (e.g. a Flask app where a user enters their ID or a movie they liked)

## Expected performance
SVD collaborative filtering on this dataset typically achieves **RMSE ≈ 0.87-0.90** (on the 0.5-5.0 rating scale). Note collaborative filtering can't recommend movies with zero ratings — the content-based approach fills that gap.

## For your report
Worth discussing: why collaborative filtering captures "people who liked X also liked Y" patterns invisible to content features (e.g. two movies with wildly different genres that share an audience), versus why content-based filtering is more explainable and interpretable ("recommended because it's also Animation/Comedy") — and why real production systems (Netflix, Myntra's product recs, etc.) almost always use hybrid approaches for exactly the reasons your comparison here demonstrates.
