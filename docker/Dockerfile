# Reproducible image for the p-factor multimodal notebook (JupyterLab + full scientific stack).
# Build context is this docker/ folder; the repo is mounted at runtime by docker-compose.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    NILEARN_DATA=/project/src/data/nilearn_cache

# Minimal system deps for building wheels and for nilearn/matplotlib rendering.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ca-certificates \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

# Register a kernel named nh2026 so the notebook's kernelspec resolves inside the container.
RUN python -m ipykernel install --name nh2026 --display-name "Python (nh2026)"

WORKDIR /project/src/jupiter_notebooks
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--ServerApp.token=", "--ServerApp.password="]
