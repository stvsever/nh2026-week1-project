"""Anonymous, cached access to the open RBC PNC dataset.

``rbclib`` resolves ``rbc://`` paths to the public ``fcp-indi`` S3 bucket. Annexed
files (the actual data) are fetched directly over HTTPS from S3 (anonymous); small
non-annexed files (``participants.tsv``) come straight from git via ``rbclib``. Every
download is cached to disk so re-runs are free.
"""
from __future__ import annotations
import io, time, urllib.request, warnings
from urllib.error import HTTPError
from pathlib import Path
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)   # quiet google/cloudpathlib py3.9 notices
import rbclib

# Stateless resolver (reads GITHUB_TOKEN from the environment if present).
_R = rbclib.RBCClient()

TARGET = "p_factor_mcelroy_harmonized_all_samples"
# Sibling psychopathology factors - never use as features (leakage / withheld for test).
LEAK = ["internalizing_mcelroy_harmonized_all_samples",
        "externalizing_mcelroy_harmonized_all_samples",
        "attention_mcelroy_harmonized_all_samples"]

_FS = "PNC_FreeSurfer"
_XCPD = "PNC_XCP-D"
_PARTICIPANTS_URL = "rbc://PNC_BIDS/study-PNC_desc-participants.tsv"


def _s3_bytes(url: str, retries: int = 3) -> bytes:
    """Resolve the git-annex pointer (1 GitHub request) then GET the bytes (1 S3
    request). ~4x faster than cloudpathlib's per-open metadata round-trips."""
    for attempt in range(retries):
        try:
            try:
                s3 = _R._get_s3_path(url)                 # annexed -> 's3://bucket/key'
            except rbclib.RBCFileException as e:          # non-annexed: bytes already fetched
                return e.contents
            bucket, _, key = s3[len("s3://"):].partition("/")
            req = urllib.request.Request(f"https://{bucket}.s3.amazonaws.com/{key}",
                                         headers={"User-Agent": "Mozilla/5.0"})
            return urllib.request.urlopen(req, timeout=60).read()
        except HTTPError as e:
            if e.code in (403, 404) or attempt == retries - 1:
                raise
            time.sleep(1 + attempt)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1 + attempt)


def load_tsv(url: str, cache_path: str | Path | None = None, index_col=None) -> pd.DataFrame:
    """Load a TSV from an ``rbc://`` URL, using ``cache_path`` on disk if given."""
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            return pd.read_csv(cache_path, sep="\t", index_col=index_col)
    data = _s3_bytes(url)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(data)
    return pd.read_csv(io.BytesIO(data), sep="\t", index_col=index_col)


def subject_url(subj, kind: str) -> str:
    """``kind`` in {'brainmeasures', 'regionsurfacestats'}."""
    return f"rbc://{_FS}/freesurfer/sub-{subj}/sub-{subj}_{kind}.tsv"


def load_subject(subj, kind: str, raw_dir: str | Path) -> pd.DataFrame:
    return load_tsv(subject_url(subj, kind),
                    cache_path=Path(raw_dir) / f"sub-{subj}_{kind}.tsv")


# --- XCP-D functional derivatives (connectivity matrices, ALFF, ReHo, timeseries) ---
def xcpd_url(subj, suffix: str, ses: str = "ses-PNC1") -> str:
    """``suffix`` is everything after ``sub-<id>_<ses>_``, e.g.
    ``task-rest_acq-singleband_space-fsLR_seg-4S156Parcels_stat-pearsoncorrelation_relmat.tsv``."""
    return f"rbc://{_XCPD}/sub-{subj}/{ses}/func/sub-{subj}_{ses}_{suffix}"


def load_xcpd(subj, suffix: str, raw_dir: str | Path, ses: str = "ses-PNC1") -> pd.DataFrame:
    idx = suffix.endswith("relmat.tsv")   # connectivity matrices carry a Node index column
    return load_tsv(xcpd_url(subj, suffix, ses),
                    cache_path=Path(raw_dir) / "xcpd" / f"sub-{subj}_{suffix}",
                    index_col=0 if idx else None)


def load_xcpd_atlas(name: str, raw_dir: str | Path) -> pd.DataFrame:
    """Node-to-network label table for an XCP-D atlas, e.g. name='4S156Parcels'."""
    url = f"rbc://{_XCPD}/atlases/atlas-{name}/atlas-{name}_dseg.tsv"
    return load_tsv(url, cache_path=Path(raw_dir) / "xcpd" / f"atlas-{name}_dseg.tsv")


def load_participants(raw_dir: str | Path) -> pd.DataFrame:
    """Demographics + target for all subjects, indexed by ``participant_id``."""
    df = load_tsv(_PARTICIPANTS_URL, cache_path=Path(raw_dir) / "participants.tsv")
    return df.set_index("participant_id")
