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
leakage-free nested cross-validation: the grid is screened with a single shuffled 5-fold split, and the winning
configuration is confirmed with repeated (5x) 5-fold cross-validation. The transformer (PyTorch) uses the Apple-Metal
or CUDA GPU when present; as expected on small tabular data it does not beat the regularized linear models, which the
leaderboard reports honestly.

- **Best configuration: demographics + global structural brain, with a regularized linear model (Ridge /
  ElasticNet) or a light stacking ensemble.** Held-out **R-squared about 0.08**, predicted-observed **r about 0.29**.
- **More brain features do not help.** Adding the 102 Desikan cortical parcels or high-dimensional functional
  connectivity *lowers* held-out R-squared through overfitting. The fine-grained brain features are largely
  redundant with age (cortical thickness correlates with age at r about 0.37, stronger than age correlates with
  the p-factor), so once age and SES are in the model they add estimation noise, not new signal.
- **Functional connectivity adds little.** The 14 cortical and 16 subcortical within/between-network features
  (rest, 4S156 atlas) correlate with the p-factor at |r| < 0.08; a connectivity-only model is at the noise floor.
- **Dimensions.** The general p-factor is the *most* predictable target (R-squared about 0.08); attention is the
  best sub-dimension (about 0.06); internalizing and externalizing are near zero. The sub-dimensions are not more
  predictable than the general factor.

### Why R-squared is not higher (and that is fine)

Out-of-sample brain-based prediction of a psychopathology factor is genuinely capped low, and modern large-sample
work confirms it: [Marek et al. 2022, Nature](https://www.nature.com/articles/s41586-022-04492-9) show reproducible
brain-behavior effects are small (r about 0.1 to 0.2, needing thousands of subjects), and
[Jung et al. 2023](https://pubmed.ncbi.nlm.nih.gov/36580867/) predict the p-factor from ABCD resting connectivity
(n = 6,905) at r = 0.16 ([Xia et al. 2018, Nat Commun](https://www.nature.com/articles/s41467-018-05317-y) is the
canonical PNC study). Our r about 0.29 is at or above the top of that range and higher than the large-sample ABCD
result. R-squared 0.08 is the same result as r about 0.29 (many papers headline the correlation). R-squared above
0.3 for out-of-sample p-factor prediction is not established in the literature and would indicate leakage (e.g. the
sibling symptom scales) or overfitting. Negative R-squared for a bad tier x model cell is legitimate (worse than
predicting the mean); the search rejects those.

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
    raw/xcpd/          cached XCP-D connectivity matrices               (gitignored, large)
    processed/         pfactor_multimodal_dataset.csv (portable), leaderboards, final_model.joblib
    figures/           PNG figures written by the notebook
  results/             pfactor_test_predictions.csv (held-out p-factor predictions for submission)
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
`train_participants.tsv` / `test_participants.tsv` when they are provided. The submission file is written to
`src/results/pfactor_test_predictions.csv`. Per the challenge rules, only one group member should submit the final
predictions (open a pull request from your fork to the `results` branch of the upstream template repository).

## Run it

Easiest (Docker):

```bash
cd docker && docker compose up --build     # then open http://localhost:8888/lab
```

Or locally: open the notebook with a Python kernel whose NumPy stack is consistent (the project `.venv`,
kernel `Python (nh2026)`), and Run All. The notebook auto-detects the data (`src/data/...`), the compute device
(macOS CPU / Linux CUDA), and writes figures, a leaderboard, and the submission CSV. With `FAST_MODE = False`
(the default) it runs a strong, repeated cross-validation (about 15 to 25 minutes); set `FAST_MODE = True` for a
~2-minute screening pass. It needs internet the first time it draws the nilearn brain maps or downloads data.

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
