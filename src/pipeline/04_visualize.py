#!/usr/bin/env python
"""04 - Build the figure set from the cached results.

    data/figures/01_target_distribution.png
    data/figures/02_tier_ladder.png
    data/figures/03_model_tier_heatmap.png
    data/figures/04_model_lines.png
    data/figures/05_pred_vs_actual_best.png
    data/figures/06_learning_curves.png
    data/figures/07_feature_importance.png

Run 03_train_compare.py first.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from projectlib import features, models, plotting

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "data" / "figures"
# Honest reference band for out-of-sample brain/demographic prediction of a general
# psychopathology factor, from large-sample studies (see src/METHODS.md).
CEILING = (0.02, 0.12)


def _fit_ridge(mat, tier):
    X, y = features.get_Xy(mat, tier)
    num, cat = features.split_num_cat(list(X.columns))
    pipe = models.make_estimator(
        __import__("sklearn.linear_model", fromlist=["Ridge"]).Ridge(alpha=100.0),
        num, cat)
    pipe.fit(X, y)
    pre = pipe.named_steps["pre"]
    names = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
    coef = pipe.named_steps["est"].coef_
    return names, coef


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    mat = pd.read_parquet(PROC / "feature_matrix.parquet").set_index("participant_id")
    tier_summary = pd.read_csv(PROC / "tier_summary.csv")
    diag = json.loads((PROC / "target_diagnostics.json").read_text())
    meta = json.loads((PROC / "deployment_meta.json").read_text())

    # 1. target distribution
    plotting.target_distribution(mat["target"].dropna(), FIG / "01_target_distribution.png", diag)

    # 2. tier ladder
    plotting.tier_ladder(tier_summary, FIG / "02_tier_ladder.png", ceiling=CEILING)

    # 3 + 4. model x tier heatmap and lines
    all_cmp = pd.concat([pd.read_csv(p) for p in sorted(PROC.glob("comparison_T*.csv"))],
                        ignore_index=True)
    plotting.model_tier_heatmap(all_cmp, FIG / "03_model_tier_heatmap.png", value="test_R2")
    plotting.model_lines(all_cmp, FIG / "04_model_lines.png", value="test_R2")

    # 5. prediction vs actual (best/deployment tier)
    preds = pd.read_csv(PROC / "best_predictions.csv")
    plotting.pred_vs_actual(preds["p_factor_actual"], preds["p_factor_pred"],
                            FIG / "05_pred_vs_actual_best.png",
                            title=f"Held-out predictions: {meta['tier_label']} / {meta['model']}",
                            floor=diag["min"])

    # 6. learning curves (train vs CV) for two representative algorithms on the best tier
    from sklearn.model_selection import learning_curve
    from sklearn.linear_model import Ridge
    from xgboost import XGBRegressor
    X, y = features.get_Xy(mat, meta["chosen_tier"])
    num, cat = features.split_num_cat(list(X.columns))
    curves = {}
    for name, est in [("Ridge", Ridge(alpha=100.0)),
                      ("XGBoost", XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                               subsample=0.8, colsample_bytree=0.7,
                                               reg_lambda=10.0, n_jobs=1, verbosity=0))]:
        pipe = models.make_estimator(est, num, cat)
        sizes, tr, va = learning_curve(pipe, X, y, cv=5, scoring="r2", n_jobs=-1,
                                       train_sizes=np.linspace(0.15, 1.0, 6))
        curves[name] = (sizes, tr.mean(1), tr.std(1), va.mean(1), va.std(1))
    plotting.learning_curves(curves, FIG / "06_learning_curves.png")

    # 7. feature importance (Ridge coefficients on the best tier)
    names, coef = _fit_ridge(mat, meta["chosen_tier"])
    plotting.feature_importance(names, coef, FIG / "07_feature_importance.png",
                                title=f"Top Ridge coefficients ({meta['tier_label']})")

    # 8. SHAP importance for the deployment model (model-agnostic, on transformed features)
    _shap_importance(mat, meta, FIG / "08_shap_importance.png")
    print(f"[04] wrote 8 figures -> {FIG}")


def _shap_importance(mat, meta, path):
    """Mean absolute SHAP value per feature for the saved deployment model."""
    import shap, joblib
    from sklearn.compose import TransformedTargetRegressor
    bundle = joblib.load(PROC / "deployment_model.joblib")
    model, cols = bundle["model"], bundle["feature_columns"]
    X, _ = features.get_Xy(mat, meta["chosen_tier"])
    X = X.reindex(columns=cols)
    pipe = model.regressor_ if isinstance(model, TransformedTargetRegressor) else model
    pre, est = pipe.named_steps["pre"], pipe.named_steps["est"]
    Xt = pre.transform(X)
    fnames = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
    rng = np.random.RandomState(0)
    samp = rng.choice(len(Xt), size=min(300, len(Xt)), replace=False)
    Xs = Xt[samp]
    try:
        if est.__class__.__name__ in ("XGBRegressor", "RandomForestRegressor",
                                      "HistGradientBoostingRegressor"):
            sv = shap.TreeExplainer(est).shap_values(Xs)
        else:
            bg = Xt[rng.choice(len(Xt), size=min(80, len(Xt)), replace=False)]
            sv = shap.Explainer(est.predict, bg)(Xs).values
        mean_abs = np.abs(np.asarray(sv)).mean(axis=0)
        plotting.shap_importance(fnames, mean_abs, path,
                                 title=f"SHAP importance: {meta['tier_label']} / {meta['model']}")
    except Exception as e:                                             # noqa: BLE001
        print(f"[04] SHAP skipped ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
