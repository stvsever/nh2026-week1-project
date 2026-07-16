# Data

All data here comes from the **open-source** [Reproducible Brain Charts (RBC)](
https://reprobrainchart.github.io/) release of the **Philadelphia Neurodevelopmental
Cohort (PNC)** - the same dataset the project targets. Nothing is private; everything
is fetched programmatically and cached locally.

## Where it comes from

| Source | What | How we read it |
|---|---|---|
| [`ReproBrainChart/PNC_FreeSurfer`](https://github.com/ReproBrainChart/PNC_FreeSurfer) | preprocessed FreeSurfer morphometry (per subject) | `rbclib` → public S3 |
| [`ReproBrainChart/PNC_XCP-D`](https://github.com/ReproBrainChart/PNC_XCP-D) | parcelwise functional connectivity, ALFF, ReHo (per subject) | `rbclib` → public S3 |
| [`ReproBrainChart/PNC_BIDS`](https://github.com/ReproBrainChart/PNC_BIDS) | participant demographics + `p_factor` | `rbclib` → GitHub |
| S3 bucket `s3://fcp-indi/data/Projects/RBC/` | the actual annexed data blobs | **anonymous** (`no_sign_request`) HTTPS GET |

`rbclib` (from `Neurohackademy2026/rbclib`) maps `rbc://…` paths to that public S3
bucket. Access is anonymous - no AWS credentials. Off the NeuroHackademy Hub, set a
GitHub token to avoid GitHub's anonymous rate limit:

```bash
export GITHUB_TOKEN=$(gh auth token)
```

## The per-subject files we use (all already preprocessed, all small)

| File | Size | Contents |
|---|---|---|
| `sub-<id>_brainmeasures.tsv` | **~12 KB** | global + subcortical (aseg) volumes, hemispheric cortical thickness/area, QC (Euler number) |
| `sub-<id>_regionsurfacestats.tsv` | **~2.8 MB** | cortical morphometry for ~30 atlases (Desikan `aparc`, Destrieux, Schaefer, Yeo networks, Glasser, …) × metrics (`ThickAvg`, `SurfArea`, `GrayVol`, …) |
| `xcpd/…_stat-pearsoncorrelation_relmat.tsv` | **~0.5 MB** | 156×156 Pearson functional connectivity matrix (4S156 atlas: 100 Schaefer cortical + 56 subcortical/cerebellar), for rest and both tasks |
| `xcpd/…_stat-{alff,reho}_bold.tsv` | **~5 KB** | parcelwise ALFF / ReHo, for rest and both tasks |

We deliberately **never** download the large `*.tar.xz` FreeSurfer archives or the 4D
BOLD from `PNC_CPAC`. Everything above is small, per subject, and cached; the full set
is about 6 GB. Parallel S3 throughput is roughly 10 MB/s (the older ~0.27 MB/s note was
a pessimistic Hub measurement).

## Target: `p_factor`

`p_factor_mcelroy_harmonized_all_samples` - a harmonized general-psychopathology
factor ([McElroy et al.](https://doi.org/10.1101/2025.02.24.639850)). Continuous
regression target, ~standardized, with a **floor near −1.6** and mild right skew
(see `data/figures/03_target_distribution.png`).

### ⚠️ Integrity / no test labels locally
The provided `train_participants.tsv` / `test_participants.tsv` live **only on the
Hub**, and the test `p_factor` is withheld (`NaN`). Locally we therefore hold out our
own random **test split** from the labeled subjects (tuning by CV on the train part
only); the official test set is never touched. The
raw RBC participants file *does* contain every subject's true `p_factor`; we use it
**only** to supply training labels and demographic features, never to fill test
predictions, and we never use the sibling factors
(`internalizing/externalizing/attention`) as inputs.

## Folder layout

```
data/
  raw/        # cached FreeSurfer TSVs + participants.tsv + manifest.json  (gitignored)
  raw/xcpd/   # cached XCP-D connectivity / ALFF / ReHo TSVs + atlas labels  (gitignored)
  processed/  # feature_matrix.parquet, comparison tables, deployment model, metrics
  figures/    # tier ladder, model heatmap, learning curves, prediction-vs-actual, importance
```

Regenerate everything with the numbered scripts in [`../src/pipeline/`](../src/pipeline).
