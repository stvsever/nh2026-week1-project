# Pipeline

Numbered, sequential scripts. Each imports the `projectlib` package (in `../projectlib`)
and reads or writes under `../../data/`. Run them in order. Full rationale is in
[`../METHODS.md`](../METHODS.md).

```bash
export GITHUB_TOKEN=$(gh auth token)          # off-Hub only (avoids GitHub rate limits)

python src/pipeline/01_fetch_data.py          # -> data/raw/        FreeSurfer TSVs (about 4.3 GB)
python src/pipeline/01b_fetch_xcpd.py          # -> data/raw/xcpd/   XCP-D connectivity/ALFF/ReHo (about 1.7 GB)
python src/pipeline/02_build_features.py       # -> data/processed/feature_matrix.parquet
python src/pipeline/03_train_compare.py        # -> comparison_*.csv, tier_summary.csv, deployment_model.joblib
python src/pipeline/04_visualize.py            # -> data/figures/*.png
python src/pipeline/05_predict.py --participants test_participants.tsv   # official test set
```

| Step | Script | Does |
|---|---|---|
| 01 | `01_fetch_data.py` | download the two small FreeSurfer TSVs for every labeled subject (cached, resumable, hard capped by `--max-gb`) |
| 01b | `01b_fetch_xcpd.py` | download XCP-D parcelwise functional connectivity, ALFF, and ReHo (4S156 atlas) for rest and both tasks |
| 02 | `02_build_features.py` | assemble one wide, namespaced feature matrix and cache it as parquet |
| 03 | `03_train_compare.py` | compare the model zoo across eight modality tiers, pick the best by CV R squared, refit and save the deployment model |
| 04 | `04_visualize.py` | tier ladder, model x tier heatmap, learning curves, prediction vs actual, feature importance |
| 05 | `05_predict.py` | fill `p_factor` for a withheld participants table using the saved model |

Everything runs locally, about 6 GB of cached derivatives total.

## Methodology (why the results are trustworthy)

- Eight modality tiers, from demographics through structural MRI, functional connectivity,
  a full multimodal model, and a curated theory-driven set, isolate the marginal value of
  each modality.
- Fixed held out test split, stratified on target quantiles, never seen during tuning.
- Tuning by cross validated R squared on the training split only; models and
  hyperparameters are selected by CV, then scored once on the held out test set.
- No leakage: impute, scale, one hot, variance threshold, and the target transform all
  live inside the pipeline and are refit within every fold. eTIV normalization is a per
  subject transform.
- No label leakage: sibling factors (internalizing, externalizing, attention) and the
  acquisition batch are never used as features; motion and Euler QC are exclusion or
  nuisance only.
- Primary metric R squared, with MAE, RMSE, and Pearson r reported alongside a mean
  baseline.
