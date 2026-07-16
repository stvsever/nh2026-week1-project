#!/usr/bin/env python
"""01b - Fetch XCP-D functional derivatives (connectivity, ALFF, ReHo) into data/raw/xcpd.

XCP-D ships small per-subject TSVs: a parcelwise Pearson functional connectivity matrix
(``*_stat-pearsoncorrelation_relmat.tsv``) and parcelwise ALFF and ReHo, for the resting
run and the two tasks. We use the 4S156 atlas (100 Schaefer cortical parcels + 56
subcortical and cerebellar parcels, each labeled with a Yeo 7/17 network), which keeps
the connectivity matrices small (~0.5 MB) while covering cortex and subcortex.

    export GITHUB_TOKEN=$(gh auth token)
    python src/pipeline/01b_fetch_xcpd.py               # all labeled subjects
"""
from __future__ import annotations
import sys, json, time, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from projectlib import dataio

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
XCPD = RAW / "xcpd"

SEG = "4S156Parcels"
# (task token, connectivity stat suffixes) - rest uses the singleband acquisition tag.
TASKS = {
    "rest":     "task-rest_acq-singleband",
    "frac2back": "task-frac2back",
    "idemo":    "task-idemo",
}
# per-task suffixes to fetch (relmat = FC matrix; alff/reho = parcelwise amplitude/homogeneity)
SUFFIXES = {
    "relmat": f"space-fsLR_seg-{SEG}_stat-pearsoncorrelation_relmat.tsv",
    "alff":   f"space-fsLR_seg-{SEG}_stat-alff_bold.tsv",
    "reho":   f"space-fsLR_seg-{SEG}_stat-reho_bold.tsv",
}


def _fetch(subj, task_tag, kind, suffix):
    fname = f"{task_tag}_{suffix}"
    path = XCPD / f"sub-{subj}_{fname}"
    if path.exists() and path.stat().st_size > 0:
        return True, 0
    try:
        dataio.load_xcpd(subj, fname, RAW)
        return True, path.stat().st_size if path.exists() else 0
    except Exception as e:                                             # noqa: BLE001
        return False, type(e).__name__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="cap subjects (0 = all labeled)")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--kinds", nargs="+", default=["relmat", "alff", "reho"])
    a = ap.parse_args()

    XCPD.mkdir(parents=True, exist_ok=True)
    demo = dataio.load_participants(RAW)
    subs = demo[demo[dataio.TARGET].notna()].index.tolist()
    if a.n > 0:
        subs = subs[:a.n]

    # cache atlas label table once
    dataio.load_xcpd_atlas(SEG, RAW)
    print(f"[01b] cached atlas dseg for {SEG}")

    jobs = [(s, TASKS[t], k, SUFFIXES[k])
            for s in subs for t in a.tasks for k in a.kinds]
    print(f"[01b] {len(subs)} subjects x {len(a.tasks)} tasks x {len(a.kinds)} kinds "
          f"= {len(jobs)} files", flush=True)

    ok, fail, done, t0 = 0, {}, 0, time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_fetch, *j): j for j in jobs}
        for fut in as_completed(futs):
            good, info = fut.result()
            done += 1
            if good:
                ok += 1
            else:
                fail[str(info)] = fail.get(str(info), 0) + 1
            if done % 500 == 0 or done == len(jobs):
                rate = done / max(time.time() - t0, 1e-9)
                print(f"[01b]   {done}/{len(jobs)}  ok={ok} fail={sum(fail.values())} "
                      f"({rate:.0f}/s, {(RAW/'xcpd').stat().st_size if False else ''})", flush=True)
    print(f"[01b] failures: {fail}")
    (XCPD / "manifest_xcpd.json").write_text(json.dumps(
        {"seg": SEG, "tasks": a.tasks, "kinds": a.kinds, "n_subjects": len(subs),
         "ok": ok, "fail": fail}, indent=2))
    print(f"[01b] done -> {XCPD}")


if __name__ == "__main__":
    main()
