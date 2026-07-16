#!/usr/bin/env python
"""05 - Predict p_factor for a new (withheld) participants table with the saved model.

Usage (when NeuroHackademy releases ``test_participants.tsv`` with NaN p_factor)::

    export GITHUB_TOKEN=$(gh auth token)
    python src/pipeline/05_predict.py --participants /path/to/test_participants.tsv \
        --out data/processed/test_predictions.tsv

The script fetches each test subject's cached FreeSurfer TSVs if missing, builds the
exact feature columns the deployment model expects, predicts, and writes the table
with the ``p_factor`` column filled in. Nothing about the test labels is ever used.
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from projectlib import dataio, features

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", required=True, help="TSV with a participant_id column")
    ap.add_argument("--out", default=str(PROC / "test_predictions.tsv"))
    ap.add_argument("--model", default=str(PROC / "deployment_model.joblib"))
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    bundle = joblib.load(a.model)
    model, cols = bundle["model"], bundle["feature_columns"]
    print(f"[05] loaded deployment model (tier={bundle['tier']}, {len(cols)} features)")

    demo = pd.read_csv(a.participants, sep="\t").set_index("participant_id")
    subs = demo.index.tolist()
    print(f"[05] {len(subs)} test subjects; fetching FreeSurfer TSVs if missing...")
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for kind in ("brainmeasures", "regionsurfacestats"):
            list(ex.map(lambda s: _safe(dataio.load_subject, s, kind), subs))

    # the RBC participants file carries demographics; join if the test table lacks them
    ref = dataio.load_participants(RAW)
    for c in features.DEMO_NUM + features.DEMO_CAT:
        if c not in demo.columns:
            demo[c] = ref.reindex(demo.index)[c]
    if dataio.TARGET not in demo.columns:
        demo[dataio.TARGET] = np.nan

    # Build brain features without QC dropping (every test subject must be predicted).
    needs_regions = any(c.startswith(("cortex:", "net:", "parcel:")) for c in cols)
    mat = features.build_matrix(subs, demo, RAW, with_regions=needs_regions,
                                n_workers=a.workers, euler_min=None, verbose=True)
    # Align to ALL requested subjects: brain columns are NaN where a scan is missing
    # (the pipeline imputes them); demographics are always present.
    X = mat.reindex(demo.index)
    for c in features.DEMO_NUM + features.DEMO_CAT:
        X[f"demo:{c}"] = demo[c].values
    X = features.add_derived(X).reindex(columns=cols)      # exact training column order
    pred = model.predict(X)

    out = demo.copy()
    out[dataio.TARGET] = pred
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.reset_index().to_csv(a.out, sep="\t", index=False)
    print(f"[05] wrote {len(out)} predictions -> {a.out}  "
          f"(brain features for {len(mat)} subjects, demographics for all {len(out)})")


def _safe(fn, *args):
    try:
        fn(*args, RAW)
    except Exception:
        pass


if __name__ == "__main__":
    main()
