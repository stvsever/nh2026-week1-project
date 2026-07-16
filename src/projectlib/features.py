"""Feature engineering for ``p_factor`` prediction from RBC PNC FreeSurfer data.

The module assembles ONE wide, namespaced per-subject feature matrix from the two
cached TSVs and then exposes six *nested* feature tiers of increasing complexity.
Building the master matrix once and slicing it per tier keeps everything consistent
(same subjects, same imputation, same QC) across the whole model comparison.

Column namespaces (prefix before the first ``:``)
    demo:*        demographics (age, sex, race, ethnicity, handedness, educations, bmi)
    global:*      global structural MRI (eTIV, tissue-class volumes, mean thickness, ...)
    subcort:*     subcortical volumes, bilateral mean and left-right asymmetry
    cortex:*      Desikan (aparc) cortical morphometry, bilateral means
    net:*         functional-network *morphometry* (Yeo 7/17, Schaefer-100 networks)
    parcel:*      fine Schaefer-200 parcel morphometry (high dimensional; full tier only)
    qc:*          quality-control metrics (FreeSurfer Euler number) - NOT a predictor
    target        p_factor
    _etiv         head size, kept only for normalization/inspection

Volumes and surface areas are expressed as a fraction of estimated total intracranial
volume (eTIV) so that they measure regional composition rather than head/body size;
cortical thickness is left in millimetres because it is largely head-size independent.
This normalization is a per-subject transform, so it introduces no train/test leakage.

See ``src/METHODS.md`` for the scientific rationale of every choice here.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from . import dataio

# --------------------------------------------------------------------------- #
#  Demographics
# --------------------------------------------------------------------------- #
DEMO_NUM = ["age", "bmi"]
DEMO_CAT = ["sex", "race", "ethnicity", "handedness",
            "participant_education", "parent_1_education", "parent_2_education"]

# --------------------------------------------------------------------------- #
#  brainmeasures.tsv (global + subcortical + QC)
# --------------------------------------------------------------------------- #
# Bilateral subcortical structures present as Left_/Right_*_Volume_mm3.
SUBCORT_BILAT = ["Thalamus_Proper", "Caudate", "Putamen", "Pallidum", "Hippocampus",
                 "Amygdala", "Accumbens_area", "VentralDC", "Cerebellum_Cortex",
                 "Cerebellum_White_Matter", "Lateral_Ventricle"]
SUBCORT_MIDLINE = ["Brain_Stem", "Third_Ventricle", "Fourth_Ventricle", "CSF",
                   "CC_Anterior", "CC_Central", "CC_Posterior"]

# Global tissue-class / whole-brain volumes (raw column -> short name).
GLOBAL_VOL = {
    "TotalGray_TotalGrayVol":                     "total_gray",
    "Cortex_CortexVol":                           "cortex_vol",
    "CerebralWhiteMatter_CerebralWhiteMatterVol": "cerebral_wm",
    "SubCortGray_SubCortGrayVol":                 "subcort_gray",
    "SupraTentorial_SupraTentorialVol":           "supratentorial",
    "BrainSegNotVent_BrainSegVolNotVent":         "brainseg_notvent",
}
EULER_MIN = -250   # QC: drop scans with a worse (more negative) FreeSurfer Euler number

# --------------------------------------------------------------------------- #
#  regionsurfacestats.tsv (cortical + functional-network morphometry)
# --------------------------------------------------------------------------- #
APARC = "aparc"                                             # Desikan, 34 regions/hemi
YEO7 = "Yeo2011_7Networks_N1000"
YEO17 = "Yeo2011_17Networks_N1000"
SCHAEFER100 = "Schaefer2018_100Parcels_7Networks_order"
SCHAEFER200 = "Schaefer2018_200Parcels_7Networks_order"
SKIP_STRUCT = {"???", "unknown", "Background+FreeSurfer_Defined_Medial_Wall",
               "FreeSurfer_Defined_Medial_Wall", "Medial_Wall"}
_ATLASES_NEEDED = [APARC, YEO7, YEO17, SCHAEFER100, SCHAEFER200]


# --------------------------------------------------------------------------- #
#  XCP-D functional connectivity (4S156 atlas: 100 Schaefer + 56 subcortical/cerebellar)
# --------------------------------------------------------------------------- #
FC_SEG = "4S156Parcels"
FC_TASKS = {"rest": "task-rest_acq-singleband", "frac2back": "task-frac2back",
            "idemo": "task-idemo"}
_FC_ATLAS_CACHE: dict = {}


def _fc_groups(raw_dir):
    """Return (group_of_node list, ordered unique groups). 7 Yeo cortical networks plus
    Subcort, Hipp, Amyg, Cereb for the non-cortical 4S parcels."""
    if "g" in _FC_ATLAS_CACHE:
        return _FC_ATLAS_CACHE["g"], _FC_ATLAS_CACHE["G"]
    atlas = dataio.load_xcpd_atlas(FC_SEG, raw_dir)
    labels = atlas["label"].tolist()
    net = atlas["network_label"].fillna("NA").tolist()

    def grp(i):
        if net[i] != "NA":
            return net[i]
        l = labels[i]
        if "Cerebell" in l:
            return "Cereb"
        if "Hippocampus" in l:
            return "Hipp"
        if "Amygdala" in l:
            return "Amyg"
        return "Subcort"
    groups = [grp(i) for i in range(len(labels))]
    order = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default",
             "Subcort", "Hipp", "Amyg", "Cereb"]
    G = [g for g in order if g in set(groups)]
    _FC_ATLAS_CACHE["g"], _FC_ATLAS_CACHE["G"] = groups, G
    _FC_ATLAS_CACHE["labels"] = labels
    return groups, G


def _fc_network_summary(z, groups, G):
    """Within/between group mean Fisher-z FC from a node-by-node z matrix."""
    out = {}
    gidx = {g: [i for i in range(len(groups)) if groups[i] == g] for g in G}
    for a in range(len(G)):
        for b in range(a, len(G)):
            ia, ib = gidx[G[a]], gidx[G[b]]
            block = z[np.ix_(ia, ib)]
            if a == b:
                if len(ia) > 1:
                    vals = block[np.triu_indices(len(ia), k=1)]
                else:
                    vals = np.array([np.nan])
            else:
                vals = block.ravel()
            out[f"{G[a]}-{G[b]}"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
    return out


def _xcpd_cached(subj, suffix, raw_dir):
    """Return the cache path if the XCP-D file was already downloaded, else None (so the
    feature build never triggers a network fetch for a genuinely absent run)."""
    p = Path(raw_dir) / "xcpd" / f"sub-{subj}_{suffix}"
    return p if p.exists() and p.stat().st_size > 0 else None


def _fc_row(subj, raw_dir):
    """Functional-network FC (within/between + global) and ALFF/ReHo network means for
    each available task. Reads only cached files. Returns a namespaced dict or None."""
    groups, G = _fc_groups(raw_dir)
    gidx = {g: [i for i in range(len(groups)) if groups[i] == g] for g in G}
    out = {"participant_id": subj}
    got = False
    for task, tag in FC_TASKS.items():
        suf = f"{tag}_space-fsLR_seg-{FC_SEG}_stat-pearsoncorrelation_relmat.tsv"
        p = _xcpd_cached(subj, suf, raw_dir)
        if p is not None:
            try:
                m = pd.read_csv(p, sep="\t", index_col=0).values.astype(float)
                z = np.arctanh(np.clip(m, -0.999, 0.999))
                np.fill_diagonal(z, 0.0)
                iu = np.triu_indices(z.shape[0], k=1)
                for k, v in _fc_network_summary(z, groups, G).items():
                    out[f"fc:{task}:{k}"] = v
                out[f"fc:{task}:global"] = float(np.nanmean(z[iu]))
                got = True
            except Exception:
                pass
        for stat, ns in (("alff", "alff"), ("reho", "reho")):
            suf = f"{tag}_space-fsLR_seg-{FC_SEG}_stat-{stat}_bold.tsv"
            p = _xcpd_cached(subj, suf, raw_dir)
            if p is None:
                continue
            try:
                s = pd.read_csv(p, sep="\t").iloc[0].values.astype(float)
                for g in G:
                    out[f"{ns}:{task}:{g}"] = float(np.nanmean(s[gidx[g]]))
                got = True
            except Exception:
                pass
    return out if got else None


def _etiv(bm: pd.Series) -> float:
    for c in ("EstimatedTotalIntraCranialVol_eTIV_lh",
              "EstimatedTotalIntraCranialVol_eTIV_rh",
              "EstimatedTotalIntraCranialVol_eTIV"):
        v = bm.get(c)
        if v is not None and np.isfinite(v) and v > 0:
            return float(v)
    return np.nan


# --------------------------------------------------------------------------- #
#  Per-subject row assembly
# --------------------------------------------------------------------------- #
def _brainmeasures_row(subj, raw_dir) -> dict | None:
    try:
        bm = dataio.load_subject(subj, "brainmeasures", raw_dir).iloc[0]
    except Exception:
        return None
    etiv = _etiv(bm)
    if not np.isfinite(etiv):
        return None
    row = {"participant_id": subj, "_etiv": etiv}

    # global: eTIV (log) + tissue-class volume fractions + cortical mean thickness/area
    row["global:etiv_log"] = np.log(etiv)
    for raw, short in GLOBAL_VOL.items():
        v = bm.get(raw)
        row[f"global:{short}_frac"] = (v / etiv) if v is not None and np.isfinite(v) else np.nan
    th = [bm.get("Cortex_MeanThickness_lh"), bm.get("Cortex_MeanThickness_rh")]
    row["global:mean_thickness"] = np.nanmean(th) if np.isfinite(np.nanmean(th)) else np.nan
    area = [bm.get("Cortex_PialSurfArea_lh"), bm.get("Cortex_PialSurfArea_rh")]
    row["global:pial_area_frac"] = np.nansum(area) / etiv if np.isfinite(np.nansum(area)) else np.nan

    # subcortical: bilateral mean fraction + left-right asymmetry index
    for s in SUBCORT_BILAT:
        l, r = bm.get(f"Left_{s}_Volume_mm3"), bm.get(f"Right_{s}_Volume_mm3")
        if l is not None and r is not None and np.isfinite(l) and np.isfinite(r):
            row[f"subcort:{s}_mean_frac"] = 0.5 * (l + r) / etiv
            row[f"subcort:{s}_asym"] = (l - r) / (l + r) if (l + r) > 0 else 0.0
        else:
            row[f"subcort:{s}_mean_frac"] = np.nan
            row[f"subcort:{s}_asym"] = np.nan
    for s in SUBCORT_MIDLINE:
        v = bm.get(f"{s}_Volume_mm3")
        row[f"subcort:{s}_frac"] = (v / etiv) if v is not None and np.isfinite(v) else np.nan

    # QC
    row["qc:euler"] = np.nanmean([bm.get("lh_euler"), bm.get("rh_euler")])
    return row


def _region_row(subj, raw_dir) -> dict | None:
    try:
        rs = dataio.load_subject(subj, "regionsurfacestats", raw_dir)
    except Exception:
        return None
    rs = rs[rs.atlas.isin(_ATLASES_NEEDED) & ~rs.StructName.isin(SKIP_STRUCT)]
    etiv = np.nan  # region file has no eTIV; volumes normalized later by the brainmeasures eTIV
    out = {"participant_id": subj}

    # Desikan aparc: bilateral means of thickness, area, gray volume (per 34 regions).
    ap = rs[rs.atlas == APARC]
    for name, grp in ap.groupby("StructName"):
        out[f"cortex:aparc:{name}:thick"] = grp["ThickAvg"].mean()
        out[f"cortex:aparc:{name}:area"] = grp["SurfArea"].sum()
        out[f"cortex:aparc:{name}:gvol"] = grp["GrayVol"].sum()

    # Functional-network morphometry: per Yeo/Schaefer network, mean thickness and
    # total area/gray volume (bilateral). Structural summaries inside network parcels.
    for atlas, tag in ((YEO7, "yeo7"), (YEO17, "yeo17"), (SCHAEFER100, "sch100")):
        sub = rs[rs.atlas == atlas]
        if atlas == SCHAEFER100:
            net = sub["StructName"].str.extract(r"7Networks_(?:LH|RH)_([A-Za-z]+)")[0]
        else:
            net = sub["StructName"]  # Yeo StructName is already the network id
        sub = sub.assign(_net=net.values)
        for nname, grp in sub.groupby("_net"):
            out[f"net:{tag}:{nname}:thick"] = np.average(grp["ThickAvg"], weights=grp["NumVert"]) \
                if grp["NumVert"].sum() > 0 else grp["ThickAvg"].mean()
            out[f"net:{tag}:{nname}:area"] = grp["SurfArea"].sum()
            out[f"net:{tag}:{nname}:gvol"] = grp["GrayVol"].sum()

    # Fine Schaefer-200 parcel morphometry (thickness + gray volume) for the full tier.
    s2 = rs[rs.atlas == SCHAEFER200]
    for _, r in s2.iterrows():
        p = f"{r.hemisphere}_{r.StructName}"
        out[f"parcel:sch200:{p}:thick"] = r["ThickAvg"]
        out[f"parcel:sch200:{p}:gvol"] = r["GrayVol"]
    return out


# --------------------------------------------------------------------------- #
#  Master matrix
# --------------------------------------------------------------------------- #
def build_matrix(subjects, demo, raw_dir, with_regions=True, with_fc=True, n_workers=12,
                 euler_min=EULER_MIN, verbose=True) -> pd.DataFrame:
    """Assemble the wide namespaced feature matrix for ``subjects``.

    Rows without a usable brainmeasures file (or failing the Euler QC threshold) are
    dropped. Region- and FC-derived columns are NaN for subjects lacking those files;
    downstream tiers simply have fewer complete subjects for those columns.
    """
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        base_rows = [r for r in ex.map(lambda s: _brainmeasures_row(s, raw_dir), subjects) if r]
    if verbose:
        print(f"[feat] brainmeasures parsed for {len(base_rows)}/{len(subjects)} subjects")
    base = pd.DataFrame(base_rows).set_index("participant_id")

    if with_regions:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            reg_rows = [r for r in ex.map(lambda s: _region_row(s, raw_dir), base.index) if r]
        if verbose:
            print(f"[feat] regionsurfacestats parsed for {len(reg_rows)}/{len(base)} subjects")
        reg = pd.DataFrame(reg_rows).set_index("participant_id")
        # normalize region volumes/areas by eTIV (per-subject; leakage-free)
        etiv = base["_etiv"]
        for c in reg.columns:
            if c.endswith(":area") or c.endswith(":gvol"):
                reg[c] = reg[c] / reg.index.map(etiv)
        mat = base.join(reg, how="left")
    else:
        mat = base

    if with_fc:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            fc_rows = [r for r in ex.map(lambda s: _fc_row(s, raw_dir), base.index) if r]
        if verbose:
            print(f"[feat] XCP-D functional connectivity parsed for {len(fc_rows)}/{len(base)} subjects")
        if fc_rows:
            fc = pd.DataFrame(fc_rows).set_index("participant_id")
            mat = mat.join(fc, how="left")

    # attach demographics + target (single concat -> no frame fragmentation)
    d = demo.reindex(mat.index)
    demo_block = pd.DataFrame({f"demo:{c}": d[c].values for c in DEMO_NUM + DEMO_CAT},
                              index=mat.index)
    demo_block["target"] = d[dataio.TARGET].values
    mat = pd.concat([mat, demo_block], axis=1).copy()

    # QC filter on Euler number (euler_min=None disables it, e.g. when predicting a
    # test set where every subject must receive a prediction)
    if euler_min is not None and "qc:euler" in mat.columns:
        keep = mat["qc:euler"].fillna(0) >= euler_min
        if verbose:
            print(f"[feat] Euler QC (>= {euler_min}) keeps {int(keep.sum())}/{len(mat)}")
        mat = mat[keep]
    return mat


# --------------------------------------------------------------------------- #
#  Feature tiers: a modality ladder from demographics to full multimodal, plus a
#  curated theory-driven tier. Prefix-based tiers are defined by the namespaces they
#  include; the theory tier is an explicit, literature-grounded column selection.
# --------------------------------------------------------------------------- #
TIER_PREFIXES = {
    "T1_demographics":   ["demo:"],
    "T2_global":         ["demo:", "global:"],
    "T3_subcortical":    ["demo:", "global:", "subcort:"],
    "T4_cortical_aparc": ["demo:", "global:", "subcort:", "cortex:aparc:"],
    "T5_structural_all": ["demo:", "global:", "subcort:", "cortex:aparc:", "net:", "parcel:"],
    "T6_functional":     ["demo:", "fc:", "alff:", "reho:"],
    "T7_multimodal":     ["demo:", "global:", "subcort:", "cortex:aparc:", "net:",
                          "fc:", "alff:", "reho:"],
    "T8_theory":         None,   # explicit column list (see theory_columns)
}
TIER_LABELS = {
    "T1_demographics":   "1. Demographics",
    "T2_global":         "2. + Global structural",
    "T3_subcortical":    "3. + Subcortical",
    "T4_cortical_aparc": "4. + Cortical (Desikan)",
    "T5_structural_all": "5. All structural",
    "T6_functional":     "6. Functional (FC + ALFF/ReHo)",
    "T7_multimodal":     "7. Multimodal (struct + func)",
    "T8_theory":         "8. Theory-driven",
}

# Literature-grounded curated features for the theory tier. Patterns are matched
# against available columns, so missing ones are silently skipped.
THEORY_PATTERNS = [
    # basic demographic / socioeconomic correlates of psychopathology
    "demo:age", "demo:sex", "demo:bmi", "demo:parent_1_education", "demo:parent_2_education",
    # limbic / stress-circuit structure (hippocampus, amygdala, accumbens)
    "subcort:Hippocampus_mean_frac", "subcort:Amygdala_mean_frac",
    "subcort:Accumbens_area_mean_frac", "subcort:Hippocampus_asym", "subcort:Amygdala_asym",
    "global:total_gray_frac", "global:mean_thickness",
    # default-mode and cingulo-opercular cortical thickness
    "cortex:aparc:medialorbitofrontal:thick", "cortex:aparc:posteriorcingulate:thick",
    "cortex:aparc:isthmuscingulate:thick", "cortex:aparc:rostralanteriorcingulate:thick",
    "cortex:aparc:precuneus:thick", "cortex:aparc:superiorfrontal:thick",
    "cortex:aparc:insula:thick",
    # functional connectivity of the circuits implicated in the p-factor
    # (pair keys follow the fixed group order Vis..Default, Subcort, Hipp, Amyg, Cereb)
    "fc:rest:Default-Default", "fc:idemo:Default-Default", "fc:idemo:Limbic-Default",
    "fc:idemo:Default-Amyg", "fc:idemo:SalVentAttn-Amyg", "fc:idemo:Cont-Default",
    "fc:rest:global", "fc:idemo:global", "fc:frac2back:Cont-Cont",
    "reho:rest:Default", "alff:rest:Default", "alff:idemo:Amyg",
]


def theory_columns(mat: pd.DataFrame) -> list[str]:
    return [c for c in THEORY_PATTERNS if c in mat.columns]


def add_derived(X: pd.DataFrame) -> pd.DataFrame:
    """Add nonlinear age terms (age, age^2) used by every tier that has demographics."""
    X = X.copy()
    if "demo:age" in X.columns:
        X["demo:age2"] = X["demo:age"] ** 2
    return X


def tier_columns(mat: pd.DataFrame, tier: str) -> list[str]:
    if tier == "T8_theory":
        return theory_columns(mat)
    prefixes = TIER_PREFIXES[tier]
    return [c for c in mat.columns
            if any(c.startswith(p) for p in prefixes) and c != "target"]


def split_num_cat(cols) -> tuple[list[str], list[str]]:
    cat = [c for c in cols if c.replace("demo:", "") in DEMO_CAT]
    num = [c for c in cols if c not in cat]
    return num, cat


def get_Xy(mat: pd.DataFrame, tier: str, require_complete_regions=True):
    """Return (X, y) for a tier, restricted to its complete-case subjects.

    Structural region columns (cortex/net/parcel) must be fully present; for functional
    tiers we require the subject to have at least one FC/ALFF/ReHo value (missing tasks
    are median-imputed inside the model pipeline)."""
    cols = tier_columns(mat, tier)
    X = add_derived(mat[cols])
    y = mat["target"].astype(float)
    ok = y.notna()
    reg_cols = [c for c in cols if c.startswith(("cortex:", "net:", "parcel:"))]
    if require_complete_regions and reg_cols:
        ok = ok & X[reg_cols].notna().all(axis=1)
    func_cols = [c for c in cols if c.startswith(("fc:", "alff:", "reho:"))]
    if func_cols:
        ok = ok & X[func_cols].notna().any(axis=1)
    return X[ok], y[ok]


def target_diagnostics(y) -> dict:
    y = pd.Series(y).dropna()
    return {"n": int(y.size), "mean": float(y.mean()), "std": float(y.std()),
            "min": float(y.min()), "max": float(y.max()), "skew": float(y.skew()),
            "frac_at_floor": float((y <= y.min() + 1e-6).mean())}
