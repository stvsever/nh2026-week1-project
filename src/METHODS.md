# Methods: predicting the general psychopathology factor (p_factor) from RBC PNC data

This document records the full methodology of the pipeline in `src/` and justifies each
choice at an academic standard. It is organized top down: data, feature engineering,
modeling protocol, results, and an honest discussion of how much variance is actually
predictable. There are no em dashes anywhere in this file by request.

The target is the harmonized general psychopathology factor `p_factor`
(`p_factor_mcelroy_harmonized_all_samples`) for participants of the Philadelphia
Neurodevelopmental Cohort (PNC), released openly through Reproducible Brain Charts
(RBC). The scientific question is: how much of the between-person variance in `p_factor`
can be predicted, out of sample, from openly available demographics, FreeSurfer
morphometry, and functional connectivity, and which feature families and algorithms do
best.

---

## 1. Design summary

1. Assemble one wide, namespaced per-subject feature matrix from cached FreeSurfer and
   XCP-D derivatives plus demographics.
2. Define a modality ladder of feature tiers, from demographics only up to a full
   multimodal model, plus a curated theory-driven tier.
3. For every tier, tune a zoo of regressors with cross validated R squared on a training
   split, then score the cross validation winner on a fixed held out test set.
4. Select the single best (tier, model) by cross validated R squared, refit it on all
   labeled subjects, and save it for the withheld official test set.
5. Report R squared as the primary metric, with MAE, RMSE, and Pearson r alongside.

The whole pipeline runs locally in a few minutes of modeling after a one time data
download of about 6 GB. Nothing is uploaded and no test labels are ever used.

---

## 2. Data

### 2.1 Source and access

All data come from the open RBC release of PNC. Access is anonymous: `rbclib` resolves
`rbc://` paths to the public `fcp-indi` S3 bucket, and the annexed data blobs are fetched
over HTTPS with no credentials. Every download is cached under `data/raw/` so re runs are
free. See `src/projectlib/dataio.py`. We use three RBC derivative repositories:

| Repository | What we take | Size on disk |
|---|---|---|
| `PNC_FreeSurfer` | `brainmeasures.tsv` and `regionsurfacestats.tsv` per subject | about 4.3 GB |
| `PNC_XCP-D` | parcelwise functional connectivity, ALFF, ReHo per subject | about 1.7 GB |
| `PNC_BIDS` | the participants table (demographics + target) | tiny |

### 2.2 Cohort and sample

The RBC participants table lists 1601 PNC participants, each at a single wave, of whom
1600 have a non missing `p_factor`. Of these, 1591 have both FreeSurfer TSVs and 1480
have at least one XCP-D functional run. After the FreeSurfer quality control filter
(section 3.11) the structural modeling sample is 1499 subjects; the functional tiers use
the subset of those with functional data.

### 2.3 Target: p_factor

`p_factor` is a continuous, approximately standardized general psychopathology factor.
McElroy and colleagues (https://doi.org/10.1101/2025.02.24.639850) harmonized
psychopathology across the RBC studies by expert 1-to-1 semantic matching of clinical
items (GOASSESS in PNC, the Child Behavior Checklist elsewhere) down to 22 shared items,
then fit a bifactor model. The general factor of that model is `p_factor`; the specific
factors are the orthogonal internalizing, externalizing, and attention factors. In this
cohort `p_factor` has mean near -0.45, standard deviation near 0.94, a hard floor near
-1.61 (a large minority of subjects sit at or near this floor), and mild right skew. The
floor and skew motivate comparing the raw target against a Yeo-Johnson transform during
modeling. The distribution is in `data/figures/01_target_distribution.png`.

### 2.4 Predictors used

- Demographics: age, sex, race, ethnicity, handedness, participant education, both parent
  education variables, and body mass index (BMI).
- FreeSurfer morphometry: global tissue class volumes, subcortical volumes, Desikan
  cortical morphometry, and network parcellated cortical morphometry.
- XCP-D functional connectivity: parcelwise Pearson connectivity for the resting run and
  the two tasks (fractional 2-back and emotion identification), plus parcelwise ALFF and
  ReHo.

### 2.5 What we deliberately excluded, and why

- Sibling psychopathology factors. `internalizing`, `externalizing`, and `attention` (the
  orthogonal specific factors from the same bifactor model) are never used as features.
  They share item content with the general factor, they are withheld for the official
  test subjects, and using them would be leakage.
- Acquisition group (`cubids_acquisition_group`). This encodes scanner and acquisition
  batch, not biology. Using it as a predictor would let the model exploit site confounds.
- Motion and data quality (framewise displacement, DVARS, FreeSurfer Euler number).
  Motion can correlate with psychopathology, but predicting `p_factor` through data
  quality is scientifically misleading and brittle. We use the Euler number strictly as
  an exclusion criterion, never as a predictor.
- Constant columns (`study`, `study_site`, `session_id`, `wave`) carry no information.

### 2.6 Functional connectivity: what is available and what we did

An earlier version of this project incorrectly concluded that functional connectivity was
out of scope. That was wrong, and the correction is worth stating plainly. The
`PNC_CPAC` repository ships only 4D BOLD and nuisance regressors, but the separate
`PNC_XCP-D` repository ships fully post-processed, parcelwise derivatives as small TSVs:

- `stat-pearsoncorrelation_relmat.tsv`: a node by node Pearson functional connectivity
  matrix, for the resting run and both tasks, at many atlas resolutions.
- `stat-mean_timeseries.tsv`: parcel mean time series.
- `stat-alff_bold.tsv` and `stat-reho_bold.tsv`: parcelwise amplitude of low frequency
  fluctuations and regional homogeneity.

We used the 4S156 atlas (100 Schaefer cortical parcels labeled with Yeo 7 and 17 network
membership, plus 56 subcortical and cerebellar parcels including explicit hippocampus,
amygdala, accumbens, thalamic, and cerebellar nodes). We downloaded the connectivity
matrices and the ALFF and ReHo maps for the resting run, the fractional 2-back task, and
the emotion identification task, for every labeled subject that has them (about 1480
subjects, about 1.7 GB). Section 3.9 describes the FC features, section 5 the results, and
section 6 the finding that connectivity adds essentially no out of sample signal here.

---

## 3. Feature engineering

All feature construction lives in `src/projectlib/features.py`. It builds one wide matrix
whose columns are namespaced by the prefix before the first colon, then slices that matrix
into tiers. Building once guarantees that every tier and every model sees exactly the same
subjects, imputation, and quality control.

### 3.1 Namespaces

| Prefix | Meaning |
|---|---|
| `demo:` | demographics |
| `global:` | global structural MRI |
| `subcort:` | subcortical volumes (bilateral mean and asymmetry) |
| `cortex:` | Desikan (aparc) cortical morphometry |
| `net:` | functional network structural morphometry (Yeo 7/17, Schaefer 100) |
| `parcel:` | fine Schaefer 200 parcel morphometry |
| `fc:` | functional connectivity, within and between network mean Fisher-z |
| `alff:`, `reho:` | ALFF and ReHo, network mean |
| `qc:` | quality control (Euler number), not a predictor |

### 3.2 Head size normalization

Regional volumes and surface areas are expressed as a fraction of estimated total
intracranial volume (eTIV); cortical thickness is left in millimetres because it is
largely head-size independent. eTIV itself is kept (log transformed). This is a per
subject transform, so it is leakage free.

### 3.3 Demographics

Age, BMI (numeric), and sex, race, ethnicity, handedness, participant education, and both
parent education variables (categorical, one hot encoded inside the model pipeline). A
quadratic age term captures the strong nonlinear neurodevelopmental trajectory (ages 8 to
23). Age is the strongest single predictor (r about +0.22), then BMI and parent education.

### 3.4 Global structural

Log eTIV; total gray, cortex, cerebral white matter, subcortical gray, supratentorial, and
non ventricular brain volumes as eTIV fractions; mean cortical thickness; total pial
surface area as an eTIV fraction.

### 3.5 Subcortical

For each bilateral aseg structure (thalamus, caudate, putamen, pallidum, hippocampus,
amygdala, accumbens, ventral diencephalon, cerebellum, lateral ventricle): a bilateral
mean eTIV fraction and a left minus right asymmetry index. Midline structures enter as
single eTIV fractions.

### 3.6 Cortical morphometry (Desikan)

Per Desikan region (34 per hemisphere): bilateral mean thickness, bilateral summed surface
area (eTIV fraction), and bilateral summed gray volume (eTIV fraction).

### 3.7 Functional network structural morphometry

Per Yeo 7, Yeo 17, and Schaefer 100 network: vertex weighted mean thickness, total surface
area, and total gray volume (bilateral). This is structural morphometry labeled by
functional network atlases, not connectivity.

### 3.8 Fine parcel morphometry

Schaefer 200 per parcel thickness and gray volume (eTIV fraction), included in the full
structural tier to test whether fine spatial detail plus regularization beats the compact
tiers.

### 3.9 Functional connectivity, ALFF, and ReHo

Each connectivity matrix is Fisher z transformed. The 156 nodes are grouped into 11
functional groups: the 7 Yeo cortical networks plus subcortical, hippocampus, amygdala,
and cerebellum. For each task (rest, 2-back, emotion) we compute the mean within group and
between group connectivity (a compact 11 by 11 summary, 66 values per task) plus the global
mean connectivity. ALFF and ReHo are summarized as the mean over each of the 11 groups per
task. This yields a low dimensional, interpretable functional feature set that directly
encodes within-network integration and between-network segregation, including default mode
and limbic circuits implicated in psychopathology. We also evaluated the full edge level
connectivity (all 12090 edges) separately with heavy regularization and connectome-based
predictive modeling (section 6).

### 3.10 The feature tiers

Each structural tier is a strict superset of the previous one; the functional, multimodal,
and theory tiers add or curate across modalities.

| Tier | Name | Contents |
|---|---|---|
| T1 | Demographics | demographics + age squared |
| T2 | + Global structural | + global volumes, thickness, area |
| T3 | + Subcortical | + subcortical mean and asymmetry |
| T4 | + Cortical (Desikan) | + 34 region thickness, area, volume |
| T5 | All structural | + network morphometry and Schaefer 200 parcels |
| T6 | Functional | demographics + FC + ALFF + ReHo (rest and both tasks) |
| T7 | Multimodal | all structural + all functional |
| T8 | Theory-driven | curated cross-modal set (see below) |

The theory-driven tier (T8) is a small, literature-grounded selection: age, sex, BMI, and
parent education; hippocampus, amygdala, and accumbens volume and asymmetry; total gray
volume and mean thickness; default mode and cingulo-opercular cortical thickness (medial
orbitofrontal, posterior and anterior cingulate, precuneus, superior frontal, insula); and
default mode, limbic, and amygdala functional connectivity from the emotion task and rest,
plus global connectivity and default mode ALFF and ReHo. It tests whether a focused,
theory-first feature set outperforms the broad tiers by avoiding noise.

### 3.11 Quality control

Subjects whose mean FreeSurfer Euler number is worse than -250 are dropped (a standard
surface reconstruction threshold), removing 92 subjects (1591 to 1499). The Euler number
is used only for exclusion, never as a predictor.

---

## 4. Modeling

All modeling lives in `src/projectlib/models.py`.

### 4.1 Train, validation, and test protocol

For each tier a fixed 20 percent test set is held out once, stratified on target quantile
bins. On the 80 percent training set every model is tuned with `RandomizedSearchCV` using
5 fold cross validation (the validation signal). The cross validation winner is refit on
the whole training set and scored once on the untouched test set. This is a clean train,
validation (the cross validation folds), test separation.

### 4.2 Leakage control

Every preprocessing step (imputation, standardization, variance thresholding, one hot
encoding, and the optional target transform) lives inside the scikit-learn pipeline, so it
is refit within every cross validation fold on training data only. The eTIV normalization
and the FC Fisher z transform are per subject, so they are leakage free by construction.

### 4.3 Model zoo

Baseline (predict the training mean), Ridge, ElasticNet, Support Vector Regression (RBF),
Kernel Ridge (RBF), Random Forest, Histogram Gradient Boosting, XGBoost, a multilayer
perceptron with early stopping and L2 weight decay, and a stacking fusion (a Ridge meta
learner over Ridge, ElasticNet, SVR, and Histogram Gradient Boosting base learners trained
on 5 fold out of fold predictions). Search widths are trimmed on high dimensional tiers to
keep the whole comparison inside a few minutes.

### 4.4 Target transform

Because `p_factor` has a floor and mild skew, every model is run on the raw target and with
a Yeo-Johnson transform whose predictions are inverse transformed, so all metrics are in
`p_factor` units. The raw target performs at least as well, so the deployed model uses no
transform.

### 4.5 Model selection and metrics

The winner within a tier, and the deployment model across tiers, are chosen by cross
validated R squared (an estimate of generalization), never by the held out test score, so
the test metrics stay honest. R squared is the primary metric, with MAE, RMSE, and Pearson
r reported alongside. Reporting r as well as R squared matters: a model can have a clearly
positive predicted to actual correlation while its R squared is small, because predictions
are correctly shrunk toward the mean.

### 4.6 Deployment to the official test set

The chosen model is refit on all labeled subjects and saved to
`data/processed/deployment_model.joblib` with its exact feature column list.
`src/pipeline/05_predict.py` then fills `p_factor` for a withheld participants table,
fetching any missing derivatives, rebuilding the exact columns, and imputing missing brain
data so every test subject receives a prediction. No test label is read at any point.

---

## 5. Results

Filled from the latest full run (`data/processed/metrics.json`, `tier_summary.csv`,
`deployment_meta.json`) and the figures in `data/figures/`.

<!-- RESULTS_AUTOFILL -->

### 5.4 Functional connectivity contributes essentially nothing

Beyond the tier table above, we evaluated functional connectivity directly and
exhaustively, because it is the modality most often hoped to carry psychopathology signal:

- Resting network level connectivity (the 66 within and between network summaries):
  cross validated R squared near or below zero, r about 0.00 to 0.05.
- Resting edge level connectivity (all 12090 edges) with Ridge across a wide range of
  regularization, with PCA, and with connectome-based predictive modeling (select the
  edges most correlated with the target inside cross validation): best r about 0.05 to
  0.10, R squared negative.
- Task connectivity: the emotion identification task is the best of any functional
  modality at r about 0.17 with R squared near 0.01; the 2-back task is weaker.
- ALFF and ReHo: no improvement.
- Adding functional connectivity to demographics does not raise, and often lowers, cross
  validated R squared.

This is fully consistent with the literature (section 6). Functional connectivity is
included in the tiers for completeness and honesty, not because it helps.

Figures:

- `01_target_distribution.png`: `p_factor` distribution, floor, and Yeo-Johnson version.
- `02_tier_ladder.png`: cross validation and test R squared across the modality ladder.
- `03_model_tier_heatmap.png`: test R squared for every model and tier.
- `04_model_lines.png`: each algorithm as a line across tiers.
- `05_pred_vs_actual_best.png`: held out predictions of the deployment model.
- `06_learning_curves.png`: train versus cross validation R squared against sample size.
- `07_feature_importance.png`: top coefficients on the deployment tier by feature family.

---

## 6. How much variance is actually predictable, and the R squared target

The stated aspiration was a test R squared above 0.40. It is important to be honest that
this is not attainable from these demographic and brain features, and that any pipeline
reporting it on this data is almost certainly exploiting leakage or optimistic evaluation
rather than real generalizable signal. Three independent lines of evidence set the ceiling.

1. Empirical, this project. Demographics alone reach cross validated R squared about 0.08,
   with age the strongest predictor. No structural tier improves on that meaningfully, and
   no functional tier (rest, task, network, or all 12090 edges) exceeds r about 0.17 with
   R squared near zero, evaluated with proper nested cross validation.

2. Literature. Out of sample prediction of a general psychopathology factor from resting
   connectivity reaches only r about 0.16 even in ABCD with roughly 7000 subjects across
   many sites, and a broad review concluded that all brain-based models of psychopathology
   yield out of sample predictions with r squared below 0.15. Brain-behavior association
   effect sizes are small and require very large samples to estimate reliably (Marek and
   colleagues, 2022, Nature).

3. Construct. `p_factor` is a latent factor over clinical symptom items. The features that
   would predict it at R squared 0.40 are other measures of the same symptoms, not brain
   morphometry or connectivity in a sample of about 1500.

A realistic honest ceiling here is roughly R squared 0.02 to 0.12, shown as a shaded band
in `02_tier_ladder.png`. Our models land inside that band. A reported R squared above 0.40
on this data is best explained by one of: using the raw psychopathology items or the
sibling internalizing, externalizing, and attention factors (which share the general
factor) as inputs, which is leakage; predicting into the training data or with non nested
cross validation, which is optimistic; or a small favorable test split. The value of this
project is a rigorous, leakage free, multimodal, and interpretable estimate of the small
real effect, and a deployment model chosen to generalize to the withheld official test set.

---

## 7. Reproducibility

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install "cloudpathlib[s3]>=0.18.0" "git+https://github.com/Neurohackademy2026/rbclib"
export GITHUB_TOKEN=$(gh auth token)

python src/pipeline/01_fetch_data.py      # FreeSurfer TSVs (about 4.3 GB)
python src/pipeline/01b_fetch_xcpd.py      # XCP-D connectivity, ALFF, ReHo (about 1.7 GB)
python src/pipeline/02_build_features.py   # assemble and cache the wide feature matrix
python src/pipeline/03_train_compare.py    # eight tier model comparison, pick deployment model
python src/pipeline/04_visualize.py         # figures
# later, when the official test table is released:
python src/pipeline/05_predict.py --participants test_participants.tsv
```

Randomness is fixed with `seed = 0`. `analysis.ipynb` reproduces the comparison
interactively and imports the same modules, so notebook and scripts cannot diverge.

---

## 8. Limitations

- Single cohort (PNC), single wave, no cross cohort or longitudinal validation.
- Functional connectivity is short single band rest plus two tasks; single subject
  connectivity is noisy, though this does not change the conclusion.
- The `p_factor` floor and skew mean squared error based metrics are dominated by the many
  near floor subjects; we report R squared, MAE, and r together to keep this visible.
- BMI has about 30 percent missingness and is median imputed inside the pipeline.
- The held out test R squared has real sampling variability at n about 300; the cross
  validation estimate over the whole sample is the more stable summary.

---

## 9. References

- McElroy and colleagues. Harmonized psychopathology factors in RBC.
  https://doi.org/10.1101/2025.02.24.639850
- Reproducible Brain Charts. An open data resource for mapping brain development and its
  associations with mental health. Neuron, 2025.
- Marek and colleagues, 2022. Reproducible brain wide association studies require thousands
  of individuals. Nature 603, 654 to 660.
- XCP-D. A robust post-processing pipeline of fMRI data. https://xcp-d.readthedocs.io/
- Reproducible Brain Charts. https://reprobrainchart.github.io/
