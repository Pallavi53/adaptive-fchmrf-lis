# Adaptive fcHMRF-LIS Neuroimaging Demo

One-hour deployable implementation of the project:

**Adaptive fcHMRF-LIS with Deep Learning-Based ROI Detection for Neuroimaging Analysis**

## Pipeline

1. Synthetic 3D neuroimaging generation
2. Two-group voxel-wise statistical testing
3. Lightweight neural ROI detection
4. Benjamini-Hochberg FDR
5. Storey-style q-value
6. Approximate LocalFDR
7. Adaptive sparse mean-field fcHMRF-LIS
8. FDP/FNP/precision/recall/Dice/runtime evaluation
9. Interactive 3D Plotly maps
10. CSV result export

## Local run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Render

Build command:

```text
pip install -r requirements.txt
```

Start command:

```text
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

Use Python 3.13.5.

## Scope

This is a lightweight educational/research prototype. It does not reproduce every
optimization in the reference fcHMRF-LIS paper. In particular, the project uses a
sparse local approximation so that it can run on CPU-based free hosting.

The application is not a clinical diagnostic system.
