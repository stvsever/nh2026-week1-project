# Predicting the general psychopathology factor (`p_factor`) from RBC PNC

NeuroHackademy 2026, week-1 group project. We predict the harmonized general psychopathology factor
`p_factor` (and its internalizing / externalizing / attention sub-dimensions) for the Philadelphia
Neurodevelopmental Cohort (PNC), using open [Reproducible Brain Charts
(RBC)](https://reprobrainchart.github.io/) demographics, FreeSurfer structural morphometry, and XCP-D
resting-state functional connectivity, and we measure honestly how much of its variance is really predictable.

The whole analysis is one self-contained notebook:
[`src/jupiter_notebooks/pfactor_multimodal_prediction.ipynb`](src/jupiter_notebooks/pfactor_multimodal_prediction.ipynb)
(with its own [README](src/jupiter_notebooks/README.md)). It runs locally, in Docker, and on JupyterHub.

## Main findings

We search a grid of **9 feature tiers x 10 ML algorithms** (Ridge, ElasticNet, KernelRidge, SVR, RandomForest,
HistGradientBoosting, XGBoost, MLP, Stacking, and a **transformer with cross-attention for multimodal fusion**) with
leakage-free nested cross-validation, then deploy a **per-target model** chosen by training cross-validation only. The
transformer (PyTorch) uses the Apple-Metal or CUDA GPU when present; as expected on small tabular data it does not beat
the regularized linear models, which the leaderboard reports honestly.

- **Held-out test set.** The withheld challenge participants are listed in
  [`src/data/raw/held_out_set_ids.txt`](src/data/raw/held_out_set_ids.txt) (321 ids) and, on the Hub, the organizers'
  `test_participants.tsv`. They are **excluded from all training** (they carry public labels locally, so this matters),
  predicted for all four targets, and scored against the public RBC labels where available.
- **Per-target deployment models**, each selected by training cross-validation (never the held-out score): the
  p-factor and internalizing use **demographics + global structural (ElasticNet)**; externalizing and attention use
  **demographics (KernelRidge)**. Every candidate is multimodal; global brain structure genuinely helps the p-factor
  (eTIV, cortex and white-matter volume rank at the top of the permutation importance), while high-dimensional
  connectivity does not and is regularized away.
- **Missing data is handled, not dropped.** Removing high-missing features hurts (dropping bmi at about 31% missing
  lowers the p-factor cross-validated R-squared from about 0.11 to about 0.09), so bmi is kept and imputed by **KNN**
  from the ten most similar subjects (tested to beat median), with a missingness indicator; categorical gaps become
  their own "missing" category. This lifts the p-factor cross-validated R-squared to about **0.11**.
- **More brain features do not help.** Adding the 102 Desikan cortical parcels or high-dimensional functional
  connectivity *lowers* cross-validated R-squared through overfitting; the fine-grained brain features are largely
  redundant with age, so once age and SES are in the model they add estimation noise, not new signal.
- **Performance (honest).** The cross-validated score is the reliable estimate; the held-out score is a single roughly
  320-subject draw and is noisier. Both are reported.

  | target | model | CV R-squared | CV r | held-out R-squared | held-out r |
  |--------|-------|------:|-----:|------------:|-----------:|
  | p_factor | demo+global (ElasticNet) | +0.11 | +0.33 | +0.03 | +0.22 |
  | internalizing | demo+global (ElasticNet) | +0.03 | +0.18 | +0.01 | +0.12 |
  | externalizing | demo (KernelRidge) | +0.02 | +0.14 | +0.04 | +0.22 |
  | attention | demo (KernelRidge) | +0.05 | +0.23 | +0.10 | +0.35 |

  The p-factor is the most predictable in cross-validation; **attention is the standout on the held-out set**
  (R-squared about 0.10, r about 0.35). The predicted-versus-true panels also show a fitted regression slope, which is
  shallow because a well-regularized model on low-signal data shrinks its predictions (correct for R-squared).

### Why R-squared is not higher (and that is fine)

Out-of-sample brain-based prediction of a psychopathology factor is genuinely capped low, and modern large-sample
work confirms it: [Marek et al. 2022, Nature](https://www.nature.com/articles/s41586-022-04492-9) show reproducible
brain-behavior effects are small (r about 0.1 to 0.2, needing thousands of subjects), and
[Jung et al. 2023](https://pubmed.ncbi.nlm.nih.gov/36580867/) predict the p-factor from ABCD resting connectivity
(n = 6,905) at r = 0.16 ([Xia et al. 2018, Nat Commun](https://www.nature.com/articles/s41467-018-05317-y) is the
canonical PNC study). Our cross-validated r about 0.33 is at or above the top of that range and higher than the
large-sample ABCD result (many papers headline the correlation, so R-squared about 0.11 is the same result). R-squared
above 0.3 for out-of-sample p-factor prediction is not established in the literature and would indicate leakage (e.g.
the sibling symptom scales, which are withheld for the real test set too) or overfitting. Negative R-squared for a bad
tier x model cell is legitimate (worse than predicting the mean); the search rejects those.

## Repository layout

Only two folders live at the repository root: `src/` (everything) and `docker/` (reproducible container).

```
src/
  jupiter_notebooks/   pfactor_multimodal_prediction.ipynb (the deliverable) + README
  projectlib/          dataio, features, models, plotting (importable helpers, used for data fetch)
  pipeline/            01_fetch, 01b_fetch_xcpd, 02_build_features, ... (bulk data fetch on a fresh machine)
  METHODS.md           extended methodology
  data/
    raw/               cached FreeSurfer TSVs + participants.tsv        (gitignored, large)
    raw/held_out_set_ids.txt   the 321 withheld challenge participants  (committed)
    raw/xcpd/          cached XCP-D connectivity matrices               (gitignored, large)
    processed/         pfactor_multimodal_dataset.csv (portable), leaderboards, final_model.joblib
    figures/           PNG figures written by the notebook
  results/             heldout_predictions_all_targets.csv (4 targets) + per-target performance for submission
docker/                Dockerfile, requirements.txt, docker-compose.yml, .dockerignore
```

## How to use this repository (fork + JupyterHub workflow)

This repo is a fork of the NeuroHackademy 2026 template. On the JupyterHub:

```bash
git clone https://github.com/stvsever/nh2026-week1-project
cd nh2026-week1-project
```

Open [`src/jupiter_notebooks/pfactor_multimodal_prediction.ipynb`](src/jupiter_notebooks/pfactor_multimodal_prediction.ipynb)
and Run All. If the data is not already present, the notebook **downloads it automatically** from the public
ReproBrainChart mirror via `rbclib` (installed on the Hub), so a fresh clone just works. It also uses the Hub's shared
`train_participants.tsv` / `test_participants.tsv` when they are provided. The submission is written to
`src/results/heldout_predictions_all_targets.csv` (all four targets; a p-factor-only view is in
`pfactor_test_predictions.csv`). Per the challenge rules, only one group member should submit the final predictions
(open a pull request from your fork to the `results` branch of the upstream template repository).

## Run it

Easiest (Docker):

```bash
cd docker && docker compose up --build     # then open http://localhost:8888/lab
```

Or locally: open the notebook with a Python kernel whose NumPy stack is consistent (the project `.venv`,
kernel `Python (nh2026)`), and Run All. The notebook auto-detects the data (`src/data/...`), the compute device
(macOS CPU / Linux CUDA), and writes figures, a leaderboard, and the submission CSVs. With `FAST_MODE = False`
(the default) it runs a strong, repeated cross-validation with per-target model selection (about 25 to 40 minutes);
set `FAST_MODE = True` for a fast screening pass. It needs internet the first time it draws the nilearn brain maps or
downloads data.

To pre-fetch the raw data on a fresh machine (optional; the notebook does this on demand):

```bash
python src/pipeline/01_fetch_data.py     # ~4.3 GB FreeSurfer TSVs
python src/pipeline/01b_fetch_xcpd.py    # ~1.7 GB XCP-D connectivity
```

## The RBC / PNC data

Open RBC release of PNC: [`PNC_BIDS`](https://github.com/ReproBrainChart/PNC_BIDS),
[`PNC_FreeSurfer`](https://github.com/ReproBrainChart/PNC_FreeSurfer),
[`PNC_CPAC`](https://github.com/ReproBrainChart/PNC_CPAC). `p_factor` is a harmonized general psychopathology
factor ([McElroy et al.](https://doi.org/10.1101/2025.02.24.639850)). Access is anonymous through `rbclib` and the
public `fcp-indi` S3 bucket.
