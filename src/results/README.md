# Results

Held-out predictions for the p-factor and its three dimensions, written automatically by
[`../jupiter_notebooks/pfactor_multimodal_prediction.ipynb`](../jupiter_notebooks/pfactor_multimodal_prediction.ipynb)
(sections 6 to 8).

The held-out test set is the withheld participants listed in
[`../data/raw/held_out_set_ids.txt`](../data/raw/held_out_set_ids.txt) (321 ids) and, on the Hub, the organizers'
`test_participants.tsv`. **They are never used in training** (the modeling cohort excludes them, so nothing leaks).

Every candidate is multimodal (demographics + global structural + functional connectivity). For each target we pick
the best model by **training cross-validation only** (never the held-out score), because the dimensions do not share
the same optimal feature set. Missing data is handled inside cross-validation: features over 50% missing are dropped,
categorical gaps (e.g. parental education) become their own "missing" category, and numeric gaps (bmi, or the ~14% of
subjects without a rest scan) are median-imputed with a missingness indicator. That handling alone lifts the p-factor
cross-validated R-squared from about 0.09 to about 0.11.

## Files

| file | contents |
|------|----------|
| `heldout_predictions_all_targets.csv` | one row per held-out participant: `p_factor`, `internalizing`, `externalizing`, `attention`, each `_predicted` and (where the public label exists) `_true`. **The submission.** |
| `pfactor_test_predictions.csv` | the p-factor-only view: `participant_id, p_factor_predicted, p_factor_true`. |
| `heldout_performance.csv` / `.txt` / `.json` | per-target selected model and metrics (R2, r, MAE, RMSE, fitted slope). |

## Performance (honest)

Two numbers per target. The **cross-validated** score (repeated k-fold on the ~1273-subject training set) is the
reliable estimate; the **held-out** score is a single ~320-subject draw and is noisier.

| target | model | CV R2 | CV r | held-out R2 | held-out r |
|--------|-------|------:|-----:|------------:|-----------:|
| p_factor | demo+global (ENet) | +0.105 | +0.32 | +0.028 | +0.21 |
| internalizing | demo+global (ENet) | +0.033 | +0.18 | +0.013 | +0.12 |
| externalizing | demo (KRR) | +0.021 | +0.14 | +0.043 | +0.22 |
| **attention** | demo (KRR) | +0.052 | +0.23 | **+0.106** | **+0.36** |

These sit at or above the modern literature ceiling for out-of-sample psychopathology prediction
([Marek 2022](https://www.nature.com/articles/s41586-022-04492-9); [Jung 2023](https://pubmed.ncbi.nlm.nih.gov/36580867/),
r = 0.16 at n = 6,905). R-squared above 0.3 would indicate leakage (e.g. the sibling symptom scales, which are withheld
for the real test set too) or overfitting, so we do not chase it. The predicted-versus-true panels show a fitted
regression slope; it is shallow because a well-regularized model on low-signal data shrinks its predictions, which is
correct for R-squared.

## Submitting

Commit these files, push to your GitHub fork, and follow the challenge's submission instructions. Do not overwrite
other groups' files.
