#!/usr/bin/env python
"""01 - Fetch the open RBC PNC FreeSurfer derivatives into ``data/raw`` (cached).

We download the two small per-subject TSVs for every labeled subject:

* ``sub-<id>_brainmeasures.tsv``       (~12 KB)  global + subcortical + hemispheric cortex + QC
* ``sub-<id>_regionsurfacestats.tsv``  (~2.8 MB) multi-atlas cortical morphometry
                                                 (Desikan, Schaefer, Yeo networks, ...)

The large ``*.tar.xz`` FreeSurfer archives and the 4D BOLD from PNC_CPAC are never
downloaded (see ``src/METHODS.md`` for why true functional connectivity is out of
scope under the local data budget). Every download is cached, so re-runs are free.

Off the NeuroHackademy Hub, set a GitHub token first to avoid anonymous rate limits::

    export GITHUB_TOKEN=$(gh auth token)
    python src/pipeline/01_fetch_data.py                 # all labeled subjects
    python src/pipeline/01_fetch_data.py --n-full 200     # cap the heavy TSVs
"""
from __future__ import annotations
import sys, json, time, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # .../src
from projectlib import dataio

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
REGIONSTATS_MB = 2.8       # nominal size of one regionsurfacestats.tsv


def _fetch(subj, kind):
    """Return (subj, kind, ok, nbytes). Cached files count as ok with 0 new bytes."""
    path = RAW / f"sub-{subj}_{kind}.tsv"
    if path.exists() and path.stat().st_size > 0:
        return subj, kind, True, 0
    try:
        dataio.load_subject(subj, kind, RAW)
        return subj, kind, True, path.stat().st_size if path.exists() else 0
    except Exception as e:                                             # noqa: BLE001
        return subj, kind, False, f"{type(e).__name__}"


def _run(subjects, kind, workers, label):
    ok, fail, done = [], {}, 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_fetch, s, kind) for s in subjects]
        for fut in as_completed(futs):
            s, _, good, info = fut.result()
            done += 1
            (ok.append(s) if good else fail.setdefault(str(info), []).append(s))
            if done % 50 == 0 or done == len(subjects):
                rate = done / max(time.time() - t0, 1e-9)
                print(f"[01]   {label}: {done}/{len(subjects)}  ok={len(ok)} "
                      f"fail={sum(len(v) for v in fail.values())}  ({rate:.0f}/s)", flush=True)
    if fail:
        print(f"[01]   {label} failures: "
              f"{ {k: len(v) for k, v in fail.items()} }", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-base", type=int, default=0, help="cap brainmeasures subjects (0 = all labeled)")
    ap.add_argument("--n-full", type=int, default=0, help="cap regionsurfacestats subjects (0 = all labeled)")
    ap.add_argument("--max-gb", type=float, default=9.0, help="hard cap on regionsurfacestats download volume")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    print("[01] loading participants (demographics + target)...")
    demo = dataio.load_participants(RAW)
    labeled = demo[demo[dataio.TARGET].notna()]
    subs = labeled.sample(frac=1, random_state=a.seed).index.tolist()
    print(f"[01] {len(subs)} labeled subjects available")

    base_subs = subs if a.n_base <= 0 else subs[:a.n_base]
    full_cap = len(subs) if a.n_full <= 0 else a.n_full
    full_cap = min(full_cap, int(a.max_gb * 1024 / REGIONSTATS_MB))
    full_subs = subs[:full_cap]

    print(f"[01] fetching brainmeasures for {len(base_subs)} subjects (~12 KB each)...")
    ok_base = _run(base_subs, "brainmeasures", max(a.workers, 16), "brainmeasures")

    est_gb = len(full_subs) * REGIONSTATS_MB / 1024
    print(f"[01] fetching regionsurfacestats for {len(full_subs)} subjects "
          f"(~{est_gb:.1f} GB, bandwidth-bound)...")
    ok_full = _run(full_subs, "regionsurfacestats", a.workers, "regionsurfacestats")

    manifest = {"seed": a.seed, "n_labeled": len(subs),
                "base": [int(s) for s in ok_base],
                "full": [int(s) for s in ok_full]}
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[01] done -> brainmeasures={len(ok_base)} regionsurfacestats={len(ok_full)}  "
          f"manifest -> {RAW/'manifest.json'}")


if __name__ == "__main__":
    main()
