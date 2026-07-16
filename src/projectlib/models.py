"""Comparative, leakage-safe model zoo for ``p_factor`` regression.

Protocol (train / validation / test)
    1. Hold out a fixed, stratified TEST set (default 20%) that no model sees during
       tuning. Stratification is on target quantile bins so the test set spans the
       full p_factor range.
    2. On the TRAIN set, tune every model with ``RandomizedSearchCV`` (k-fold, scoring
       = R^2). The k CV folds are the VALIDATION signal used for model + hyperparameter
       selection. All preprocessing lives inside the pipeline, so it is refit within
       every fold: no leakage from validation or test into training.
    3. Refit the tuned model on the full TRAIN set and report its performance on the
       untouched TEST set.

Model selection uses cross-validated R^2 (an estimate of generalization to unseen
subjects), never the held-out TEST score, so the TEST metrics remain an honest
out-of-sample estimate. R^2 is the primary metric; MAE, RMSE and Pearson r are
reported alongside. An optional Yeo-Johnson target transform is compared against the
raw target; predictions are inverse-transformed so every metric is in p_factor units.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy.stats import loguniform, uniform, randint
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, ElasticNet, RidgeCV
from sklearn.svm import SVR, LinearSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import (RandomForestRegressor, HistGradientBoostingRegressor,
                              StackingRegressor)
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# --------------------------------------------------------------------------- #
#  Preprocessing
# --------------------------------------------------------------------------- #
def build_preprocessor(num, cat):
    """Median-impute + standardize numeric; mode-impute + one-hot categorical.

    A VarianceThreshold drops constant columns that can appear after imputation in
    small folds. Fit inside every CV fold via the enclosing pipeline (no leakage)."""
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("vt", VarianceThreshold()),
                         ("sc", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    parts = [("num", num_pipe, num)]
    if cat:
        parts.append(("cat", cat_pipe, cat))
    return ColumnTransformer(parts)


# --------------------------------------------------------------------------- #
#  Model zoo
# --------------------------------------------------------------------------- #
# Support vector machines are represented by both a linear SVM (LinearSVR) and a
# kernel SVM (SVR with an RBF kernel); kernel ridge is the closed-form kernel analog.
def zoo(n_features: int):
    """name -> (estimator, param_distributions, n_iter).

    Search widths are trimmed for high-dimensional tiers to keep the whole comparison
    inside a few minutes while still covering sensible regions of hyperparameter space.
    """
    wide = n_features > 150
    Z = {
        "Baseline (mean)": (DummyRegressor(strategy="mean"), {}, 1),
        "Ridge": (Ridge(),
                  {"est__alpha": loguniform(1e-1, 1e5)}, 30),
        "ElasticNet": (ElasticNet(max_iter=20000),
                       {"est__alpha": loguniform(1e-3, 1e2),
                        "est__l1_ratio": uniform(0.05, 0.9)}, 30),
        "SVR (linear)": (LinearSVR(max_iter=100000, random_state=0, tol=1e-3),
                         {"est__C": loguniform(1e-3, 1e1),
                          "est__epsilon": uniform(0.0, 0.3)}, 15),
        "SVR (RBF)": (SVR(kernel="rbf"),
                      {"est__C": loguniform(1e-1, 1e3),
                       "est__gamma": loguniform(1e-4, 1e0),
                       "est__epsilon": uniform(0.01, 0.4)}, 25),
        "KernelRidge (RBF)": (KernelRidge(kernel="rbf"),
                              {"est__alpha": loguniform(1e-2, 1e2),
                               "est__gamma": loguniform(1e-4, 1e0)}, 25),
        "RandomForest": (RandomForestRegressor(random_state=0, n_jobs=1),
                         {"est__n_estimators": randint(200, 500),
                          "est__max_depth": [None, 3, 5, 8],
                          "est__max_features": uniform(0.1, 0.6),
                          "est__min_samples_leaf": randint(2, 12)}, 12 if wide else 18),
        "HistGradBoost": (HistGradientBoostingRegressor(random_state=0),
                          {"est__learning_rate": loguniform(1e-2, 3e-1),
                           "est__max_depth": [None, 2, 3],
                           "est__max_leaf_nodes": randint(15, 48),
                           "est__l2_regularization": loguniform(1e-3, 1e1),
                           "est__max_iter": randint(150, 400)}, 18),
        "MLP (early-stop)": (MLPRegressor(random_state=0, early_stopping=True,
                                          validation_fraction=0.15, n_iter_no_change=15,
                                          max_iter=500),
                             {"est__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
                              "est__alpha": loguniform(1e-4, 1e1),
                              "est__learning_rate_init": loguniform(1e-4, 1e-2)}, 12),
    }
    try:
        from xgboost import XGBRegressor
        Z["XGBoost"] = (XGBRegressor(random_state=0, n_jobs=1, verbosity=0, tree_method="hist"),
                        {"est__n_estimators": randint(150, 500),
                         "est__max_depth": randint(2, 5),
                         "est__learning_rate": loguniform(1e-2, 3e-1),
                         "est__subsample": uniform(0.6, 0.4),
                         "est__colsample_bytree": uniform(0.5, 0.5),
                         "est__reg_lambda": loguniform(1e-1, 1e2),
                         "est__reg_alpha": loguniform(1e-3, 1e1)}, 18)
    except Exception:
        pass
    return Z


def _stacking_estimator(num, cat):
    """Fusion model: a Ridge meta-learner over diverse, lightly-regularized base
    learners, each a full preprocessing+model pipeline. cv=5 out-of-fold predictions
    train the meta-learner, which controls overfitting of the fusion."""
    def pipe(est):
        return Pipeline([("pre", build_preprocessor(num, cat)), ("est", est)])
    base = [
        ("ridge", pipe(Ridge(alpha=100.0))),
        ("enet", pipe(ElasticNet(alpha=0.05, l1_ratio=0.3, max_iter=20000))),
        ("svr", pipe(SVR(kernel="rbf", C=10.0, gamma=1e-3, epsilon=0.1))),
        ("hgb", pipe(HistGradientBoostingRegressor(random_state=0, learning_rate=0.05,
                                                   max_leaf_nodes=31, l2_regularization=1.0,
                                                   max_iter=300))),
    ]
    return StackingRegressor(estimators=base, final_estimator=RidgeCV(),
                             cv=5, n_jobs=-1, passthrough=False)


def make_estimator(est, num, cat, target_transform=None):
    pipe = Pipeline([("pre", build_preprocessor(num, cat)), ("est", est)])
    if target_transform == "yeojohnson":
        return TransformedTargetRegressor(regressor=pipe,
                                          transformer=PowerTransformer(method="yeo-johnson"))
    return pipe


def _prefix(dist, tt):
    return {f"regressor__{k}": v for k, v in dist.items()} if tt == "yeojohnson" else dist


def _metrics(y, p):
    return {"R2": float(r2_score(y, p)),
            "MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "r": float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 else 0.0}


# --------------------------------------------------------------------------- #
#  Per-tier comparison
# --------------------------------------------------------------------------- #
def run_tier(X, y, num, cat, test_size=0.20, cv=5, seed=0,
             transforms=("none", "yeojohnson"), include_stack=True, verbose=True):
    """Compare all models on one feature tier. Returns (table, best_info)."""
    yv = np.asarray(y, dtype=float)
    strat = pd.qcut(yv, q=min(10, max(2, len(yv) // 40)), labels=False, duplicates="drop")
    idx = np.arange(len(yv))
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=strat)
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], yv[tr], yv[te]
    icv = KFold(n_splits=cv, shuffle=True, random_state=seed)

    rows, fitted = [], {}
    models = dict(zoo(X.shape[1]))
    for tt in transforms:
        ttv = None if tt == "none" else tt
        for name, (est, dist, n_iter) in models.items():
            base = make_estimator(clone(est), num, cat, ttv)
            if dist:
                search = RandomizedSearchCV(base, _prefix(dist, tt), n_iter=n_iter, cv=icv,
                                            scoring="r2", n_jobs=-1, random_state=seed,
                                            refit=True)
                search.fit(Xtr, ytr)
                cv_r2, cv_std = search.best_score_, search.cv_results_["std_test_score"][search.best_index_]
                model = search.best_estimator_
            else:
                from sklearn.model_selection import cross_val_score
                cv_r2 = float(np.mean(cross_val_score(base, Xtr, ytr, cv=icv, scoring="r2")))
                cv_std = 0.0
                model = base.fit(Xtr, ytr)
            m = _metrics(yte, model.predict(Xte))
            rows.append({"model": name, "transform": tt, "cv_r2": float(cv_r2),
                         "cv_r2_std": float(cv_std), "test_R2": m["R2"], "test_MAE": m["MAE"],
                         "test_RMSE": m["RMSE"], "test_r": m["r"]})
            fitted[(name, tt)] = model
            if verbose:
                print(f"    {name:18s}[{tt:10s}] cv_r2={cv_r2:+.3f}  "
                      f"test R2={m['R2']:+.3f} MAE={m['MAE']:.3f} r={m['r']:+.3f}", flush=True)

    if include_stack:
        for tt in transforms:
            ttv = None if tt == "none" else tt
            stack = _stacking_estimator(num, cat)
            model = (TransformedTargetRegressor(regressor=stack,
                        transformer=PowerTransformer(method="yeo-johnson"))
                     if ttv else stack)
            from sklearn.model_selection import cross_val_score
            cv_r2 = float(np.mean(cross_val_score(model, Xtr, ytr, cv=icv, scoring="r2", n_jobs=-1)))
            model.fit(Xtr, ytr)
            m = _metrics(yte, model.predict(Xte))
            rows.append({"model": "Stacking (fusion)", "transform": tt, "cv_r2": cv_r2,
                         "cv_r2_std": 0.0, "test_R2": m["R2"], "test_MAE": m["MAE"],
                         "test_RMSE": m["RMSE"], "test_r": m["r"]})
            fitted[("Stacking (fusion)", tt)] = model
            if verbose:
                print(f"    {'Stacking (fusion)':18s}[{tt:10s}] cv_r2={cv_r2:+.3f}  "
                      f"test R2={m['R2']:+.3f} MAE={m['MAE']:.3f} r={m['r']:+.3f}", flush=True)

    table = pd.DataFrame(rows)
    # winner selected by cross-validated R^2 (generalization estimate), excluding baseline
    ranked = table[table.model != "Baseline (mean)"].sort_values("cv_r2", ascending=False)
    best = ranked.iloc[0]
    base_r2 = table.loc[table.model == "Baseline (mean)", "test_R2"].max()
    best_info = {
        "tier_n": int(len(yv)), "n_features": int(X.shape[1]),
        "best_model": best["model"], "best_transform": best["transform"],
        "cv_r2": float(best["cv_r2"]), "test_R2": float(best["test_R2"]),
        "test_MAE": float(best["test_MAE"]), "test_RMSE": float(best["test_RMSE"]),
        "test_r": float(best["test_r"]), "baseline_test_R2": float(base_r2),
        "estimator": fitted[(best["model"], best["transform"])],
        "test_idx": te, "train_idx": tr,
        "y_test": yte, "pred_test": fitted[(best["model"], best["transform"])].predict(Xte),
        "index_all": np.asarray(y.index),
    }
    return table, best_info


def fit_full(estimator_spec, X, y, num, cat):
    """Refit a chosen (model_name, transform, estimator) on ALL labeled subjects for
    deployment to the withheld official test set."""
    name, tt, fitted = estimator_spec
    model = clone(fitted)
    model.fit(X, np.asarray(y, dtype=float))
    return model
