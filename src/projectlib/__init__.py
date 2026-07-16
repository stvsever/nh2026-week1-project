"""projectlib - a small toolkit for predicting the PNC ``p_factor`` from RBC data.

Modules
-------
dataio    : anonymous access to the open RBC PNC data (via ``rbclib`` + public S3),
            with on-disk caching.
features  : one wide namespaced feature matrix (demographic, global, subcortical,
            cortical, functional-network morphometry) sliced into six nested tiers,
            plus target diagnostics.
models    : a comparative model zoo (linear, kernel, tree, neural, stacking) tuned by
            cross-validated R^2 and scored on a fixed held-out test set, leakage-safe.
plotting  : tier ladder, model-by-tier heatmap, learning curves, prediction-vs-actual,
            target distribution, and feature-importance figures.
"""
from . import dataio, features, models, plotting

__all__ = ["dataio", "features", "models", "plotting"]
