# Docker

Reproducible container for `src/jupiter_notebooks/pfactor_multimodal_prediction.ipynb` with the full stack
(numpy, pandas, scikit-learn, xgboost, shap, nilearn, jupyterlab).

## Run

```bash
cd docker
docker compose up --build
```

Then open http://localhost:8888/lab (no token needed) and run `pfactor_multimodal_prediction.ipynb`.

The whole repository is bind-mounted at `/project`, so:
- input data is read from `/project/src/data` (`src/data/raw`, `src/data/xcpd`, or the cached
  `src/data/processed/pfactor_multimodal_dataset.csv`),
- figures are written to `/project/src/data/figures`,
- the submission CSV is written to `/project/src/results`,

and everything persists on the host after the container stops.

## GPU

The image runs on CPU by default (fine for this workload; scikit-learn is CPU-only). On a Linux host with an
NVIDIA GPU and the NVIDIA Container Toolkit, uncomment the `deploy.resources` block in `docker-compose.yml`
to let XGBoost use CUDA. The notebook auto-detects the device; Apple Silicon has no supported GPU backend for
these libraries and runs on CPU.

## Files
- `Dockerfile` - Python 3.11 image, installs `requirements.txt`, registers the `nh2026` kernel, starts JupyterLab.
- `requirements.txt` - pinned-enough scientific stack.
- `docker-compose.yml` - builds the image and mounts the repo.
- `.dockerignore` - keeps the build context tiny (only `requirements.txt` ships into the image).
