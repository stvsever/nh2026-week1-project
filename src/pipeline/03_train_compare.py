#!/usr/bin/env python
"""03 - Comparative modeling across the six nested feature tiers.

For every tier the full model zoo is tuned by cross-validated R^2 on a training split
(with and without a Yeo-Johnson target transform), and the CV-selected winner is scored
on a fixed, untouched held-out test set. The single best (tier, model) by CV R^2 is then
refit on ALL labeled subjects and saved as the deployment model for the withheld official
test set. Writes:

    data/processed/comparison_<tier>.csv   all models x transforms (cv + test metrics)
    data/processed/tier_summary.csv        best model per tier
    data/processed/best_predictions.csv    held-out actual vs predicted for the best tier
    data/processed/metrics.json            machine-readable summary
    data/processed/deployment_model.joblib + deployment_meta.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from projectlib import features, models

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
SEED = 0


def load_matrix():
    df = pd.read_parquet(PROC / "feature_matrix.parquet").set_index("participant_id")
    return df


def main():
    mat = load_matrix()
    print(f"[03] matrix {mat.shape}; running 6 tiers (seed={SEED})")
    summary, tier_rows, best_global = {}, [], None

    for tier in features.TIER_PREFIXES:
        X, y = features.get_Xy(mat, tier)
        num, cat = features.split_num_cat(list(X.columns))
        print(f"\n[03] === {features.TIER_LABELS[tier]}: n={len(y)}, {X.shape[1]} features ===",
              flush=True)
        table, info = models.run_tier(X, y, num, cat, seed=SEED)
        table.insert(0, "tier", tier)
        table.to_csv(PROC / f"comparison_{tier}.csv", index=False)

        row = {"tier": tier, "label": features.TIER_LABELS[tier], "n": info["tier_n"],
               "n_features": info["n_features"], "best_model": info["best_model"],
               "transform": info["best_transform"], "cv_r2": info["cv_r2"],
               "test_R2": info["test_R2"], "test_MAE": info["test_MAE"],
               "test_RMSE": info["test_RMSE"], "test_r": info["test_r"],
               "baseline_test_R2": info["baseline_test_R2"]}
        tier_rows.append(row)
        summary[tier] = {k: v for k, v in row.items() if k not in ("tier",)}
        print(f"[03]   WINNER: {info['best_model']} [{info['best_transform']}]  "
              f"cv_r2={info['cv_r2']:+.3f}  test R2={info['test_R2']:+.3f}  "
              f"MAE={info['test_MAE']:.3f}  r={info['test_r']:+.3f}", flush=True)

        if best_global is None or info["cv_r2"] > best_global[1]["cv_r2"]:
            best_global = (tier, info, X, y, num, cat)

    pd.DataFrame(tier_rows).to_csv(PROC / "tier_summary.csv", index=False)

    # deployment: refit the CV-best (tier, model) on ALL labeled subjects of that tier
    tier, info, X, y, num, cat = best_global
    pd.DataFrame({"p_factor_actual": info["y_test"], "p_factor_pred": info["pred_test"]},
                 index=info["index_all"][info["test_idx"]]).to_csv(PROC / "best_predictions.csv")
    spec = (info["best_model"], info["best_transform"], info["estimator"])
    deploy = models.fit_full(spec, X, y, num, cat)
    joblib.dump({"model": deploy, "tier": tier,
                 "feature_columns": list(X.columns),
                 "num": num, "cat": cat}, PROC / "deployment_model.joblib")
    (PROC / "deployment_meta.json").write_text(json.dumps({
        "chosen_tier": tier, "tier_label": features.TIER_LABELS[tier],
        "model": info["best_model"], "transform": info["best_transform"],
        "cv_r2": info["cv_r2"], "test_R2": info["test_R2"], "test_r": info["test_r"],
        "n_train_all": int(len(y)), "n_features": info["n_features"]}, indent=2))

    (PROC / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[03] DEPLOYMENT: {features.TIER_LABELS[tier]} / {info['best_model']} "
          f"[{info['best_transform']}]  cv_r2={info['cv_r2']:+.3f}  test R2={info['test_R2']:+.3f}")
    print(f"[03] done -> {PROC}")


if __name__ == "__main__":
    main()
