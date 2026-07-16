#!/usr/bin/env python
"""02 - Assemble the wide feature matrix from cached TSVs and cache it.

Reads every subject that has a cached ``brainmeasures.tsv`` (region columns are added
where a ``regionsurfacestats.tsv`` is also cached), attaches demographics + target, and
writes:

    data/processed/feature_matrix.parquet    wide namespaced matrix (one row / subject)
    data/processed/tier_columns.json         column lists for each of the six tiers
    data/processed/target_diagnostics.json   distribution summary of p_factor

Run 01_fetch_data.py first.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from projectlib import dataio, features

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"


def main():
    PROC.mkdir(parents=True, exist_ok=True)
    demo = dataio.load_participants(RAW)
    labeled = demo[demo[dataio.TARGET].notna()]

    cached = sorted({p.name.split("_")[0].replace("sub-", "")
                     for p in RAW.glob("sub-*_brainmeasures.tsv")})
    subs = [s for s in labeled.index if str(s) in cached]
    print(f"[02] building matrix for {len(subs)} cached & labeled subjects...")

    mat = features.build_matrix(subs, demo, RAW, with_regions=True, n_workers=12)
    out = PROC / "feature_matrix.parquet"
    mat.reset_index().rename(columns={"index": "participant_id"}).to_parquet(out, index=False)
    print(f"[02] wrote {mat.shape} -> {out}")

    tier_cols = {t: features.tier_columns(mat, t) for t in features.TIER_PREFIXES}
    (PROC / "tier_columns.json").write_text(json.dumps(
        {t: {"n_columns": len(c), "label": features.TIER_LABELS[t]} for t, c in tier_cols.items()},
        indent=2))
    (PROC / "target_diagnostics.json").write_text(
        json.dumps(features.target_diagnostics(mat["target"]), indent=2))
    print("[02] tier sizes:", {t: len(c) for t, c in tier_cols.items()})
    print(f"[02] done -> {PROC}")


if __name__ == "__main__":
    main()
