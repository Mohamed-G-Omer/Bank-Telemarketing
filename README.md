# Bank Telemarketing — Term Deposit Subscription Prediction

A machine learning pipeline that predicts which bank customers are most likely to subscribe to a term deposit, enabling smarter and more cost-efficient telemarketing campaigns.

---

## Business Problem

A bank runs outbound telemarketing campaigns to sell term deposits. Call capacity is limited, so the goal is not to predict every subscriber perfectly — it is to **rank customers by subscription probability** so that the most promising leads are called first.

This directly answers: *"If we can only call 20% of our customer base, which 20% should we call?"*

---

## Dataset

| Property | Detail |
|---|---|
| Source | [UCI ML Repository — Bank Marketing](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) |
| File | `bank-additional-full.csv` |
| Rows | 41,188 customers |
| Features | 20 input variables + 1 target |
| Target | `y` — whether the customer subscribed ("yes" / "no") |
| Class imbalance | ~11% positive (subscribed), ~89% negative |

---

## Project Structure

```
Bank Telemarketing/
├── Bank_telemarketing.ipynb   # Full ML pipeline notebook
└── README.md
```

---

## Pipeline Overview

```
Load Data → EDA → Feature Engineering → Train/Test Split
    → Preprocessing → Cross-Validation (4 models)
        → Optuna Hyperparameter Tuning (LightGBM & CatBoost)
            → Final Model Evaluation → SHAP Explainability
                → Customer Segmentation
```

### Step-by-Step

1. **Load & Explore** — Download dataset directly from UCI; inspect shape, types, nulls, duplicates, and class balance.
2. **Feature Engineering** — Derive `pr_contacted`, `pdays_clean`, `campaign_ratio`, and `total_contacts`. Handle the 999 sentinel encoding for "never contacted."
3. **Drop Data Leakage** — Remove `duration` (call length is only known after the call ends — using it inflates ROC-AUC to ~0.94 but is useless at prediction time). Also drop raw `pdays` in favor of the cleaned version.
4. **Stratified Train/Test Split** — 80/20 split with shuffle and stratification to preserve the 11% positive rate in both sets.
5. **Preprocessing** — `ColumnTransformer` with `OneHotEncoder` for categoricals and `SimpleImputer(median)` for numericals. The same transformer is applied to all models for fair comparison.
6. **Cross-Validation Benchmark** — 5-fold stratified CV across four models using PR-AUC as the primary metric.
7. **Hyperparameter Tuning** — `Optuna` with 30 trials each for LightGBM and CatBoost. LightGBM selected as the final model.
8. **Threshold Optimization** — Optimal decision threshold found via out-of-fold predictions (F1-max), avoiding test-set leakage.
9. **Test Evaluation** — Single honest evaluation on the held-out test set.
10. **SHAP Analysis** — TreeExplainer reveals the top drivers of subscription probability.
11. **Customer Segmentation** — All customers ranked and split into three actionable tiers.

---

## Model Comparison (5-Fold CV, PR-AUC)

| Model | PR-AUC Mean | PR-AUC Std |
|---|---|---|
| Logistic Regression | 0.4374 | ±0.0087 |
| XGBoost | 0.4463 | ±0.0077 |
| LightGBM | 0.4481 | ±0.0073 |
| CatBoost | 0.4598 | ±0.0123 |

After Optuna tuning under equal conditions:
- **LightGBM tuned: 0.4711** ← selected
- CatBoost tuned: 0.4659

> All four models converge to a similar score range, suggesting ~0.47 is the natural ceiling of this dataset without the leaked `duration` feature.

---

## Final Results (Test Set — LightGBM Tuned)

| Metric | Score |
|---|---|
| ROC-AUC | **0.815** |
| PR-AUC | **0.493** |
| Lift @ top 20% | **3.34x** |
| Decision threshold | 0.204 (OOF F1-max) |

### Business Impact

| Scenario | Calls | Expected Deposits |
|---|---|---|
| Random calling (20% of test set) | 1,647 | ~186 |
| Model top-20% list | 1,647 | **~620** |
| **Additional deposits (same effort)** | — | **+434** |

Reaching the same 620 deposits via random calling would require ~5,500 calls. The model saves ~3,857 calls.

---

## Customer Segmentation Tiers

| Tier | Probability Range | Customers (test set) |
|---|---|---|
| Highly Interested — Call First | ≥ 60% | 206 |
| Medium Interest — Call with Caution | 30% – 60% | 727 |
| Low Interest — Do Not Bother | < 30% | 7,305 |

---

## Key Design Decisions

- **Why PR-AUC over Accuracy?** With 89% negatives, predicting "no" for everyone gives 89% accuracy but finds zero deposits. PR-AUC focuses entirely on the positive class.
- **Why remove `duration`?** It is post-call information. Including it causes data leakage and an artificially high ROC-AUC (~0.94).
- **Why shuffle before splitting?** The dataset is sorted by date; without shuffling, train and test fall in different time periods and the positive rate diverges between them.
- **Why OOF threshold selection?** Selecting the threshold on the test set would constitute a second form of leakage. Out-of-fold predictions on the training set are used instead.

---

## Libraries & Dependencies

```
numpy
pandas
matplotlib
seaborn
scikit-learn
lightgbm
xgboost
catboost
optuna
shap
ydata-profiling
scipy
```

Install all dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn lightgbm xgboost catboost optuna shap ydata-profiling scipy
```

---

## How to Run

1. Clone the repository and navigate to the project folder.
2. Install dependencies (see above).
3. Open `Bank_telemarketing.ipynb` in Jupyter Lab or Jupyter Notebook.
4. Run all cells top-to-bottom. The dataset is downloaded automatically from the UCI repository.

---

## Limitations & Next Steps

- The model's ceiling (~0.47 PR-AUC) is limited by what the available features can capture.
- The `duration` feature, while highly predictive, cannot be used in a realistic pre-call setting.
- **To improve further:** enrich with external data (customer transaction history, demographics, CRM notes) rather than tuning the model further.

---

## Author

Mohamed & Emre | Bank Telemarketing Project
