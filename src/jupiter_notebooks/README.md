# `pfactor_multimodal_prediction.ipynb`

One self-contained, end-to-end notebook: it predicts the general psychopathology factor (`p_factor`) and its three
sub-dimensions from PNC **demographics + structural MRI + resting-state functional connectivity**, searches a grid of
**feature tiers x ML algorithms** with leakage-free nested cross-validation, picks the configuration that maximizes
held-out R-squared, renders brain maps of the findings, and writes the submission predictions. Runs locally, in
Docker, and on JupyterHub.

## Feature tiers compared

| # | Tier | Contents |
|---|------|----------|
| 1 | demographics | age, sex, BMI, race, ethnicity, handedness, educations (one age term, no age^2) |
| 2 | + global | + global structural (eTIV, tissue-class volume fractions, mean thickness, pial area) |
| 3 | + subcortical | + aseg subcortical volumes (bilateral mean + asymmetry) |
| 4 | + cortical (full struct) | + Desikan (aparc) cortical thickness/area/gray-volume (all 34 regions) |
| 5 | theory structural | demographics + global + limbic subcortical + theory-based cortical thickness (DMN, cingulo-opercular, limbic cortex) |
| 6 | demo + FC cortical | + 14 cortical within/between Yeo-7 network connectivity |
| 7 | demo + FC all | + 14 cortical and 16 subcortical connectivity features |
| 8 | lean multimodal | theory structural (cortex + subcortex) + cortical FC |
| 9 | rich multimodal | everything (all structural + all functional) |

## ML algorithms compared

Ridge, ElasticNet, KernelRidge (RBF), SVR (RBF), RandomForest, HistGradientBoosting, XGBoost, a small MLP, a Stacking
ensemble, and a **transformer with cross-attention for multimodal fusion** (PyTorch: each modality is encoded to a
token, a transformer applies cross-modal self-attention, and a learnable fusion token cross-attends to the modalities;
it uses the Apple-Metal or CUDA GPU when available). Each self-tunes its hyperparameters by an inner cross-validation,
so the outer loop is a true nested, held-out estimate. The grid is screened with a single shuffled 5-fold split, and
the winning configuration is confirmed with repeated (5x) shuffled 5-fold cross-validation. On small tabular data the
regularized linear models match or beat the deep net, and the leaderboard shows this.

## Result (the honest ceiling)

The winner is chosen automatically: in practice **demographics + global structural with a regularized linear model or
a light stacking ensemble**.

- **p_factor: held-out R-squared about 0.08, correlation r about 0.29.**
- Adding the full cortical parcellation or high-dimensional connectivity **lowers** held-out R-squared (overfitting;
  the fine-grained brain features are largely age-redundant). The curated theory-structural tier (cortex + subcortex)
  is competitive with the demographic tiers; functional connectivity adds little.
- Dimensions (winning config): p_factor ~0.08 is the most predictable; attention ~0.06; internalizing and
  externalizing near zero. The sub-dimensions are **not** more predictable than the general factor.

Modern large-sample work confirms the ceiling: [Marek et al. 2022, Nature](https://www.nature.com/articles/s41586-022-04492-9)
(brain-behavior effects are small, r ~ 0.1 to 0.2) and [Jung et al. 2023](https://pubmed.ncbi.nlm.nih.gov/36580867/)
(p-factor from ABCD connectivity, n = 6,905, r = 0.16). Our r ~ 0.29 is at or above the top of that range. R-squared
above ~0.3 for out-of-sample p-factor prediction is not established in the literature and would indicate leakage or
overfitting.

## Brain-related findings (nilearn)

Two delta correlation maps in the exploration section: a **whole-brain volumetric mosaic** of cortical (Desikan,
mapped to Harvard-Oxford) and subcortical gray-matter volume vs p-factor, and a **bilateral glass-brain connectome**
of the 7 Yeo networks (nodes at Schaefer-derived centroids) whose edges are the between-network FC vs p-factor
correlation. Both need internet on first run (nilearn caches the atlases) and fall back to bar charts otherwise.
Every association is small (|r| < 0.1), the honest brain-behavior picture.

## Outputs

- `../data/processed/pfactor_multimodal_dataset.csv` - portable dataset (all features + 4 targets); the Hub artifact.
- `../data/figures/*.png` - target distributions, FC correlations, both brain maps, leaderboard heatmap, feature
  importance, dimension bars.
- `../data/processed/leaderboard_R2.csv`, `leaderboard_r.csv`, `dimension_results.csv`, `final_model.joblib`.
- `../results/pfactor_test_predictions.csv` - predicted p-factor for the held-out (NaN) participants (the submission).

## Data loading (adaptive)

Searches in order: the cached dataset CSV, then structural (`../data/processed/feature_matrix.parquet` or
`../data/raw` TSVs) + functional (`../data/raw/xcpd`), then `rbclib` from the public ReproBrainChart mirror. Paths are
found relatively via `find_data_root()`, so the notebook works wherever it sits under the repo.

## Uploading to https://neurohackademy.2i2c.cloud/ (without touching others' files)

1. Work only in your own `/home/jovyan/<you>/` space; never write into shared folders.
2. Push this repo to a public GitHub repo (code only; `src/data/raw` and `src/data/xcpd` are gitignored). On the Hub:
   `git clone <your-repo-url>` and open this notebook.
3. Provide the data by uploading the small `pfactor_multimodal_dataset.csv` into `src/data/processed/` (runs in
   seconds), using the Hub's shared RBC copy, or letting `rbclib` fetch it.
4. Pick the kernel and Run All. If auto-detection ever fails, set `MANUAL_ROOT` in the first cell.

## Environment

Use a consistent NumPy stack. The first cell stops with a clear message on the classic `numpy.dtype size changed`
error (wrong kernel / NumPy 1.x vs 2.x): select the project `.venv` kernel (`Python (nh2026)`), point PyCharm's
interpreter at `.venv/bin/python`, or `pip install 'numpy<2'`. `nilearn` is needed for the brain maps.
