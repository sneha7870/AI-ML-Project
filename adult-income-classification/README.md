# Adult Census Income Classification

Predicts whether a person earns `>50K` or `<=50K` per year using the UCI Adult Census Income dataset (~48,000 records).

## Approach
- **Data**: auto-downloaded from UCI ML repository (falls back to local `adult.data`/`adult.test` if you're offline — download them from the same UCI link and place in this folder).
- **EDA**: class balance, age distribution, missing value check.
- **Preprocessing**: dropped `fnlwgt` (weighting column, no predictive value) and `education` (redundant with `education_num`), label-encoded categoricals, standard-scaled numeric features.
- **Models compared**: Logistic Regression, Random Forest, Gradient Boosting.
- **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC — since the classes are imbalanced (~76% `<=50K`), F1/ROC-AUC matter more than raw accuracy.

## Run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
python train.py
```

## Outputs
- `eda_overview.png` — class + age distribution
- `model_comparison.csv` — metrics table across models
- `confusion_matrix.png`, `roc_curves.png`, `feature_importance.png`
- `best_income_model.pkl`, `scaler.pkl`, `label_encoders.pkl` — for deployment (e.g. wrap in a Flask app like your Placement Predictor project)

## Typical results
Gradient Boosting / Random Forest usually land around **85–87% accuracy** and **~0.90 ROC-AUC**; Logistic Regression is a bit lower (~82–84%) but trains instantly and is a good baseline.
