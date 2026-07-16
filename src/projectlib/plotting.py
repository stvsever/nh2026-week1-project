"""Publication-style figures for the p_factor model comparison.

Every function takes already-computed dataframes/arrays and writes a PNG. A single
model colour map and feature-namespace colour map keep the whole figure set visually
consistent. Designed to read clearly in a report: titles, axis labels, annotated
metrics, and a marked "signal ceiling" so the honest effect sizes are never oversold.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "font.size": 10.5,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
})

# consistent colours per algorithm family
MODEL_COLORS = {
    "Baseline (mean)": "#9aa0a6", "Ridge": "#1f77b4", "ElasticNet": "#4c9be8",
    "SVR (linear)": "#ffb01f", "SVR (RBF)": "#ff7f0e", "KernelRidge (RBF)": "#e6a817",
    "RandomForest": "#2ca02c", "HistGradBoost": "#17becf", "XGBoost": "#d62728",
    "MLP (early-stop)": "#9467bd", "Stacking (fusion)": "#8c564b",
}
# feature-namespace colours (for feature-importance bars)
NS_COLORS = {
    "demo": "#9467bd", "global": "#1f77b4", "subcort": "#2ca02c",
    "cortex": "#ff7f0e", "net": "#d62728", "parcel": "#8c564b", "qc": "#9aa0a6",
    "fc": "#17becf", "alff": "#e377c2", "reho": "#bcbd22",
}
NS_LABEL = {"demo": "demographics", "global": "global structural", "subcort": "subcortical",
            "cortex": "cortical (aparc)", "net": "network morphometry", "parcel": "parcels",
            "fc": "connectivity", "alff": "ALFF", "reho": "ReHo"}


def _ns(col: str) -> str:
    return col.split(":")[0]


# --------------------------------------------------------------------------- #
def tier_ladder(tier_summary, path, ceiling=None):
    """Test R^2 and CV R^2 as feature complexity grows across the six tiers."""
    ts = tier_summary.reset_index(drop=True)
    x = np.arange(len(ts))
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.plot(x, ts["cv_r2"], "o--", color="#4c9be8", lw=1.6, ms=7,
            label="cross-validation R$^2$ (train)")
    ax.plot(x, ts["test_R2"], "o-", color="#d62728", lw=2.2, ms=9,
            label="held-out test R$^2$")
    for xi, row in ts.iterrows():
        ax.annotate(row["best_model"], (xi, row["test_R2"]),
                    textcoords="offset points", xytext=(0, 11), ha="center",
                    fontsize=8, color="#333")
    ax.axhline(0, color="k", lw=0.8)
    if ceiling is not None:
        ax.axhspan(ceiling[0], ceiling[1], color="#ffd54f", alpha=0.18, zorder=0)
        ax.text(0.02, ceiling[0] + 0.001,
                "literature ceiling for brain-based p-factor prediction",
                va="bottom", ha="left", fontsize=8, color="#8a6d00",
                transform=ax.get_yaxis_transform())
    ax.set_xticks(x)
    ax.set_xticklabels(ts["label"], rotation=20, ha="right")
    ax.set_ylim(top=max(0.13, ts["cv_r2"].max() + 0.03))
    ax.set(ylabel="R$^2$ (variance explained)",
           title="p_factor is weakly predictable, and brain features add little over demographics")
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def model_tier_heatmap(all_cmp, path, value="test_R2"):
    """Heatmap of a metric across models (rows) x tiers (cols)."""
    piv = (all_cmp[all_cmp["transform"] == "none"]
           .pivot_table(index="model", columns="tier", values=value))
    # order rows by mean performance, keep baseline last
    order = piv.drop(index="Baseline (mean)", errors="ignore").mean(axis=1).sort_values(ascending=False).index.tolist()
    if "Baseline (mean)" in piv.index:
        order = order + ["Baseline (mean)"]
    piv = piv.loc[order]
    fig, ax = plt.subplots(figsize=(9.5, 6))
    vmax = np.nanmax(np.abs(piv.values))
    im = ax.imshow(piv.values, cmap="RdYlBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([c.replace("_", "\n") for c in piv.columns], fontsize=8)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.03f}", ha="center", va="center", fontsize=7.5,
                        color="k" if abs(v) < vmax * 0.6 else "w")
    ax.set_title(f"{value.replace('_',' ')} across models and feature tiers")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=value)
    ax.grid(False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def model_lines(all_cmp, path, value="test_R2"):
    """Each algorithm as a coloured line across the six tiers."""
    sub = all_cmp[all_cmp["transform"] == "none"]
    tiers = sorted(sub["tier"].unique())
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for model, g in sub.groupby("model"):
        g = g.set_index("tier").reindex(tiers)
        ax.plot(range(len(tiers)), g[value], "o-", ms=5, lw=1.8,
                color=MODEL_COLORS.get(model, "#555"),
                label=model, alpha=.5 if model == "Baseline (mean)" else .95)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels([t.replace("_", "\n") for t in tiers], fontsize=8)
    ax.set(ylabel=value.replace("_", " "),
           title="Held-out performance by algorithm and feature tier")
    ax.legend(ncol=2, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def pred_vs_actual(y, pred, path, title="Held-out predictions", floor=None):
    y = np.asarray(y); pred = np.asarray(pred)
    mae = np.mean(np.abs(y - pred))
    r = np.corrcoef(y, pred)[0, 1] if np.std(pred) > 0 else 0.0
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
    lo, hi = min(y.min(), pred.min()), max(y.max(), pred.max())
    ax[0].scatter(y, pred, s=26, alpha=.55, color="#1f77b4", edgecolor="w", linewidth=.4)
    ax[0].plot([lo, hi], [lo, hi], "k--", lw=1, label="identity")
    b, a = np.polyfit(y, pred, 1)
    xs = np.linspace(lo, hi, 50)
    ax[0].plot(xs, a + b * xs, color="#d62728", lw=1.8, label="fit")
    if floor is not None:
        ax[0].axvline(floor, color="#e0a34e", ls=":", lw=1.2, label="target floor")
    ax[0].text(.04, .96, f"R$^2$ = {r2:+.3f}\nr = {r:+.3f}\nMAE = {mae:.3f}\nn = {len(y)}",
               transform=ax[0].transAxes, va="top",
               bbox=dict(boxstyle="round", fc="#eef4ff", ec="#7aa7ff"))
    ax[0].set(xlabel="Actual p_factor", ylabel="Predicted p_factor", title=title)
    ax[0].legend(loc="lower right", fontsize=8)

    order = np.argsort(y)
    ax[1].plot(range(len(y)), y[order], "-", color="#333", lw=1.2, label="actual", alpha=.8)
    ax[1].plot(range(len(y)), pred[order], "o", ms=3, color="#d62728", label="predicted", alpha=.55)
    ax[1].set(xlabel="held-out subjects (sorted by actual)", ylabel="p_factor",
              title="Predictions track the mean, not the extremes")
    ax[1].legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def target_distribution(y, path, diag=None):
    y = np.asarray(y)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].hist(y, bins=40, color="#5b8def", edgecolor="w")
    ax[0].axvline(y.min(), color="#e0a34e", ls=":", lw=1.4, label="floor")
    ax[0].set(xlabel="p_factor", ylabel="count", title="Target distribution")
    ax[0].legend(fontsize=8)
    if diag:
        ax[0].text(.55, .97, f"n = {diag['n']}\nmean = {diag['mean']:+.2f}\n"
                   f"skew = {diag['skew']:+.2f}\nat floor = {100*diag['frac_at_floor']:.0f}%",
                   transform=ax[0].transAxes, va="top",
                   bbox=dict(boxstyle="round", fc="#fff3e0", ec="#e0a34e"))
    from sklearn.preprocessing import PowerTransformer
    yt = PowerTransformer(method="yeo-johnson").fit_transform(y.reshape(-1, 1)).ravel()
    ax[1].hist(yt, bins=40, color="#7ac77a", edgecolor="w")
    ax[1].set(xlabel="Yeo-Johnson(p_factor)", title="After Yeo-Johnson transform")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def learning_curves(curves, path):
    """curves: dict model_name -> (train_sizes, train_mean, train_std, val_mean, val_std)."""
    n = len(curves)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (name, (ts, trm, trs, vam, vas)) in zip(axes, curves.items()):
        c = MODEL_COLORS.get(name, "#1f77b4")
        ax.plot(ts, trm, "o-", color=c, label="train R$^2$")
        ax.fill_between(ts, trm - trs, trm + trs, color=c, alpha=.15)
        ax.plot(ts, vam, "s--", color="#333", label="CV R$^2$")
        ax.fill_between(ts, vam - vas, vam + vas, color="#333", alpha=.12)
        ax.axhline(0, color="k", lw=.7)
        ax.set(xlabel="training subjects", title=name)
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel("R$^2$")
    fig.suptitle("Learning curves: the train/validation gap shows overfitting control",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def shap_importance(names, mean_abs_shap, path, top=25, title="SHAP feature importance"):
    """Bar chart of mean absolute SHAP value per feature, colored by feature family."""
    idx = np.argsort(mean_abs_shap)[::-1][:top]
    names_t = [names[i] for i in idx][::-1]
    vals = np.asarray(mean_abs_shap)[idx][::-1]
    colors = [NS_COLORS.get(_ns(n), "#555") for n in names_t]
    fig, ax = plt.subplots(figsize=(9.5, 0.34 * len(names_t) + 1.5))
    ax.barh(range(len(names_t)), vals, color=colors, edgecolor="w")
    ax.set_yticks(range(len(names_t)))
    ax.set_yticklabels([n.replace("cortex:aparc:", "").replace("subcort:", "")
                        .replace("global:", "").replace("net:", "").replace("demo:", "")
                        .replace("fc:", "").replace("alff:", "ALFF ").replace("reho:", "ReHo ")
                        for n in names_t], fontsize=8)
    ax.set(xlabel="mean |SHAP value| (impact on predicted p_factor)", title=title)
    present = []
    for ns, col in NS_COLORS.items():
        if any(_ns(n) == ns for n in names_t):
            present.append(plt.Rectangle((0, 0), 1, 1, color=col, label=NS_LABEL.get(ns, ns)))
    if present:
        ax.legend(handles=present, fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def feature_importance(names, importances, path, top=25, title="Feature importance"):
    idx = np.argsort(np.abs(importances))[::-1][:top]
    names_t = [names[i] for i in idx][::-1]
    vals = np.asarray(importances)[idx][::-1]
    colors = [NS_COLORS.get(_ns(n), "#555") for n in names_t]
    fig, ax = plt.subplots(figsize=(9.5, 0.34 * len(names_t) + 1.5))
    ax.barh(range(len(names_t)), vals, color=colors, edgecolor="w")
    ax.set_yticks(range(len(names_t)))
    ax.set_yticklabels([n.replace("cortex:aparc:", "").replace("subcort:", "")
                        .replace("global:", "").replace("net:", "").replace("demo:", "")
                        for n in names_t], fontsize=8)
    ax.axvline(0, color="k", lw=.8)
    ax.set(xlabel="signed importance (standardized-feature coefficient)", title=title)
    present = []
    for ns, col in NS_COLORS.items():
        if any(_ns(n) == ns for n in names_t):
            present.append(plt.Rectangle((0, 0), 1, 1, color=col, label=NS_LABEL.get(ns, ns)))
    ax.legend(handles=present, fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
