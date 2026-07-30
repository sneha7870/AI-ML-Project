# End-to-End Render Deployment Project

A complete, tested, self-contained template for deploying a trained ML model as a live web app on Render — HTML form + JSON API + Docker + health checks, all wired together. Swap in your own model (Adult Income classifier, Student Placement Predictor, etc.) and the deployment steps stay identical.

This uses an Iris flower classifier as the demo model specifically so it needs **zero external downloads** — everything runs locally with `pip install` and stays reproducible.

## Project structure
```
render-deployment-project/
├── train_model.py       # trains + saves model.pkl and model_meta.pkl
├── app.py                # Flask app: HTML form (/) + JSON API (/predict) + health check (/health)
├── templates/index.html
├── static/style.css
├── requirements.txt
├── Dockerfile             # for Render's Docker runtime
├── Procfile               # alternative: for Render's native Python runtime (no Docker)
├── render.yaml            # Infrastructure-as-code — Render reads this automatically
└── .gitignore
```

## 1. Run locally first
```bash
pip install -r requirements.txt
python train_model.py          # creates model.pkl + model_meta.pkl
python app.py                  # runs on http://localhost:5000
```
Test the API:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```
(I ran both of these myself while building this — trains in under a second, form and API both work correctly.)

## 2. Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit: ML model deployment demo"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```
**Important**: run `train_model.py` locally and commit `model.pkl` + `model_meta.pkl` to the repo (or add a build step that trains it — see note below), since Render needs the trained artifacts to serve predictions.

## 3. Deploy on Render

### Option A — one-click via `render.yaml` (recommended)
1. Go to https://dashboard.render.com → **New** → **Blueprint**.
2. Connect your GitHub repo. Render detects `render.yaml` automatically and creates the service.
3. Click **Apply** — Render builds the Docker image and deploys.

### Option B — manual setup
1. https://dashboard.render.com → **New** → **Web Service**.
2. Connect your GitHub repo.
3. **Runtime**: Docker (it'll auto-detect the `Dockerfile`) — or choose **Python 3** and Render will use the `Procfile` instead.
4. If using the Python runtime (no Docker), set:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT app:app`
5. **Instance Type**: Free (fine for a demo/portfolio project).
6. Click **Create Web Service**. First deploy takes 2-5 minutes.

### After deploying
Render gives you a public URL like `https://ml-deployment-demo.onrender.com`. Visit it — you'll see the same form you tested locally. The `/health` endpoint is what Render's own health checks hit to confirm the service is up (configured in `render.yaml`).

## 4. Using your own model instead of Iris
1. Replace the contents of `train_model.py` with your own training pipeline (e.g. copy logic from your Adult Income or Placement Predictor projects), saving:
   - `model.pkl` — the trained model (must support `.predict()` and `.predict_proba()`)
   - `model_meta.pkl` — a dict with `feature_names` (list, in the exact order your model expects) and `class_names` (list, for classification; skip for regression and adjust `app.py` accordingly)
2. `app.py`, the HTML template, and the Dockerfile need **no changes** — they read feature/class names dynamically from `model_meta.pkl`.
3. If your model has many features (like the Adult Income dataset's 12 columns) the auto-generated form will just have more fields — still works, just less pretty. For a nicer UX at that point, consider grouping fields or defaulting less-important ones.

## Notes on the free tier
Render's free web services **spin down after 15 minutes of inactivity** and take ~30-60 seconds to wake back up on the next request — normal and expected, not a bug. Worth mentioning in your submission/demo so graders aren't confused by the first request being slow.

## What to highlight in your report
- Separation of concerns: training (`train_model.py`) is decoupled from serving (`app.py`) — you retrain offline and ship only the lightweight artifacts.
- The `/predict` JSON endpoint exists independently of the HTML form, so this backend could serve a mobile app, another service, or Postman just as easily as a browser.
- Docker gives you the exact same environment locally and in production, eliminating "works on my machine" deployment failures — worth contrasting with the `Procfile` approach where Render manages the environment for you.
