
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from scipy import ndimage, stats
from sklearn.neural_network import MLPClassifier
from statsmodels.stats.multitest import multipletests


st.set_page_config(
    page_title="Adaptive fcHMRF-LIS Neuroimaging",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Adaptive fcHMRF-LIS Neuroimaging Analyzer")
st.caption(
    "Lightweight research-demo implementation: synthetic 3D neuroimaging → "
    "neural ROI detection → sparse adaptive fcHMRF-LIS → FDR comparison → 3D maps."
)

st.warning(
    "Research/education demo only. The synthetic analysis is not a clinical diagnostic system."
)

# ---------------------------------------------------------------------
# 1. Synthetic neuroimaging data
# ---------------------------------------------------------------------
def make_spheres(shape=(24, 24, 24)):
    x, y, z = np.indices(shape)
    centers = [
        (7, 7, 7, 3.5),
        (16, 8, 14, 3.0),
        (14, 17, 9, 2.8),
    ]
    mask = np.zeros(shape, dtype=bool)
    effect = np.zeros(shape, dtype=float)
    for cx, cy, cz, r in centers:
        d2 = (x-cx)**2 + (y-cy)**2 + (z-cz)**2
        region = d2 <= r**2
        mask |= region
        # Smooth effect inside the ROI.
        effect += np.where(region, np.exp(-d2 / (2*(r/1.7)**2)), 0.0)
    if effect.max() > 0:
        effect /= effect.max()
    return mask, effect


def generate_groups(seed, n1=8, n2=8, shape=(24, 24, 24), effect_size=1.8, noise=1.0):
    rng = np.random.default_rng(seed)
    true_mask, effect = make_spheres(shape)

    g1 = rng.normal(0, noise, size=(n1, *shape))
    g2 = rng.normal(0, noise, size=(n2, *shape))

    # Disease/control group difference in the signal regions.
    for i in range(n2):
        g2[i] += effect_size * effect

    # Add a small nuisance-like spatial pattern so the task is not trivial.
    nuisance = 0.15 * ndimage.gaussian_filter(
        rng.normal(size=shape), sigma=1.2
    )
    g2 += nuisance
    return g1, g2, true_mask, effect


def voxel_statistics(g1, g2):
    t, p = stats.ttest_ind(g2, g1, axis=0, equal_var=False)
    # t -> z-like standardized statistic for this demonstration.
    z = np.sign(t) * stats.norm.isf(np.clip(p / 2.0, 1e-12, 1.0))
    z = np.nan_to_num(z)
    return t, p, z


# ---------------------------------------------------------------------
# 2. Lightweight deep-learning ROI detector
# ---------------------------------------------------------------------
def build_features(z):
    local_mean = ndimage.gaussian_filter(z, sigma=1.0)
    local_abs = ndimage.gaussian_filter(np.abs(z), sigma=1.0)

    coords = np.indices(z.shape).astype(np.float32)
    coords[0] = (coords[0] / max(z.shape[0]-1, 1)) * 2 - 1
    coords[1] = (coords[1] / max(z.shape[1]-1, 1)) * 2 - 1
    coords[2] = (coords[2] / max(z.shape[2]-1, 1)) * 2 - 1

    X = np.column_stack([
        z.ravel(),
        np.abs(z).ravel(),
        local_mean.ravel(),
        local_abs.ravel(),
        coords[0].ravel(),
        coords[1].ravel(),
        coords[2].ravel(),
    ])
    return X


@st.cache_resource
def train_roi_network(seed=42):
    rng = np.random.default_rng(seed)
    X_parts, y_parts = [], []

    # Train on independent synthetic volumes so the current test volume's
    # ground-truth mask is never passed to the ROI network.
    for k in range(3):
        s = int(seed + 100 + k)
        a, b, mask, _ = generate_groups(
            seed=s, n1=6, n2=6, shape=(20, 20, 20),
            effect_size=1.7 + 0.15*k
        )
        _, _, zz = voxel_statistics(a, b)
        X = build_features(zz)

        # Keep training manageable and approximately balanced.
        y = mask.ravel().astype(int)
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        take_pos = min(len(pos), 900)
        take_neg = min(len(neg), 1800)
        ids = np.concatenate([
            rng.choice(pos, take_pos, replace=False),
            rng.choice(neg, take_neg, replace=False),
        ])
        X_parts.append(X[ids])
        y_parts.append(y[ids])

    X_train = np.vstack(X_parts)
    y_train = np.concatenate(y_parts)

    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        max_iter=120,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=8,
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------
# 3. FDR baselines
# ---------------------------------------------------------------------
def bh_mask(p, alpha):
    flat = np.nan_to_num(p.ravel(), nan=1.0, posinf=1.0, neginf=1.0)
    reject, qvals, _, _ = multipletests(flat, alpha=alpha, method="fdr_bh")
    return reject.reshape(p.shape), qvals.reshape(p.shape)


def storey_qvalues(p, alpha):
    # Lightweight q-value approximation using Storey's pi0 estimator.
    flat = np.clip(np.nan_to_num(p.ravel(), nan=1.0), 0, 1)
    lam = 0.5
    pi0 = np.mean(flat > lam) / max(1-lam, 1e-12)
    pi0 = min(max(pi0, 0.05), 1.0)

    order = np.argsort(flat)
    ps = flat[order]
    n = len(ps)
    q = pi0 * n * ps / np.arange(1, n+1)
    q = np.minimum.accumulate(q[::-1])[::-1]

    out = np.empty_like(q)
    out[order] = q
    qmap = out.reshape(p.shape)
    return qmap <= alpha, qmap


def local_fdr_mask(z, alpha):
    # Approximate local FDR: estimate a central null density and divide
    # it by the empirical total density. This is a lightweight baseline.
    flat = z.ravel()
    null_sample = flat[np.abs(flat) <= 0.75]
    if len(null_sample) < 100:
        null_sample = flat

    grid = np.linspace(-5, 5, 801)
    bw = 0.25

    def kde(values):
        d = (grid[:, None] - values[None, :]) / bw
        return np.exp(-0.5*d*d).mean(axis=1) / (np.sqrt(2*np.pi)*bw)

    # Subsample for speed.
    rng = np.random.default_rng(123)
    vals = null_sample if len(null_sample) <= 5000 else rng.choice(null_sample, 5000, replace=False)
    null_density = kde(vals)

    vals_all = flat if len(flat) <= 8000 else rng.choice(flat, 8000, replace=False)
    total_density = kde(vals_all)

    # Interpolate densities at every voxel.
    dn = np.interp(np.clip(flat, -5, 5), grid, null_density)
    dt = np.interp(np.clip(flat, -5, 5), grid, total_density)
    lfdr = np.clip(dn / np.maximum(dt, 1e-8), 0, 1)

    return (lfdr <= alpha).reshape(z.shape), lfdr.reshape(z.shape)


# ---------------------------------------------------------------------
# 4. Sparse adaptive fcHMRF-LIS approximation
# ---------------------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def adaptive_sparse_lis(z, roi_prob, alpha=0.05, iterations=8, smoothness=0.65):
    """
    Lightweight sparse approximation of the project's adaptive fcHMRF-LIS idea.

    The full research method uses a fully connected model and efficient
    filtering. For a one-hour deployable classroom prototype, we use local
    sparse neighborhoods plus adaptive ROI probability and Gaussian filters.
    The final null probability is treated as approximate LIS.
    """
    absz = np.abs(z)
    base_signal = sigmoid((absz - 1.3) * 1.15)

    # Adaptive ROI weighting: focus the spatial inference where the neural
    # detector believes the relevant region lies.
    signal = np.clip(
        base_signal * (0.45 + 0.75 * roi_prob),
        0.0, 0.995
    )
    null_prob = 1.0 - signal

    for _ in range(iterations):
        neighbor_signal = ndimage.gaussian_filter(signal, sigma=1.0)
        # Adaptive smoothness is stronger in high ROI probability areas.
        adaptive_w = smoothness * (0.35 + 0.65 * roi_prob)
        signal = np.clip(
            (1-adaptive_w) * signal + adaptive_w * neighbor_signal,
            0.0, 0.999
        )

        # Evidence term from z-statistics.
        evidence = sigmoid((absz - 1.45) * 1.05)
        signal = np.clip(0.72 * signal + 0.28 * evidence, 0.0, 0.999)
        null_prob = 1.0 - signal

    lis = np.clip(null_prob, 0, 1)

    order = np.argsort(lis.ravel())
    sorted_lis = lis.ravel()[order]
    cumulative = np.cumsum(sorted_lis) / np.arange(1, len(sorted_lis)+1)
    valid = np.where(cumulative <= alpha)[0]

    reject = np.zeros_like(sorted_lis, dtype=bool)
    if len(valid):
        k = valid[-1] + 1
        reject[:k] = True

    mask = np.zeros_like(reject)
    mask[order] = reject
    return mask.reshape(z.shape), lis


# ---------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------
def metrics(discovery, truth):
    d = discovery.astype(bool)
    t = truth.astype(bool)

    tp = np.sum(d & t)
    fp = np.sum(d & ~t)
    fn = np.sum(~d & t)
    discoveries = np.sum(d)

    fdp = fp / max(discoveries, 1)
    fnp = fn / max(np.sum(t), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    dice = 2*tp / max(2*tp + fp + fn, 1)

    return {
        "Discoveries": int(discoveries),
        "True Positives": int(tp),
        "False Positives": int(fp),
        "FDP": float(fdp),
        "FNP": float(fnp),
        "Precision": float(precision),
        "Recall": float(recall),
        "Dice": float(dice),
    }


def surface_plot(volume, title, threshold=None):
    v = volume.copy()
    if threshold is not None:
        mask = v >= threshold
    else:
        mask = np.abs(v) >= np.percentile(np.abs(v), 97)

    coords = np.argwhere(mask)
    if len(coords) == 0:
        return go.Figure().update_layout(title=title)

    vals = v[mask]
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers",
                marker=dict(
                    size=4,
                    color=vals,
                    colorscale="Turbo",
                    opacity=0.75,
                    colorbar=dict(title="Value"),
                ),
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="cube",
        ),
        height=520,
        margin=dict(l=0, r=0, t=45, b=0),
    )
    return fig


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Experiment settings")
    seed = st.number_input("Random seed", min_value=1, max_value=99999, value=42)
    effect_size = st.slider("Abnormality strength", 0.8, 3.0, 1.8, 0.1)
    noise = st.slider("Noise level", 0.5, 2.0, 1.0, 0.1)
    alpha = st.select_slider("Target FDR (α)", options=[0.01, 0.02, 0.05, 0.10], value=0.05)
    iterations = st.slider("fcHMRF iterations", 3, 12, 8)
    smoothness = st.slider("Adaptive smoothness", 0.20, 0.90, 0.65, 0.05)

    run = st.button("🚀 Run complete analysis", type="primary", use_container_width=True)

st.info(
    "Demo pipeline: synthetic two-group 3D data → voxel-wise statistics → "
    "neural ROI probability → BH / q-value / LocalFDR → adaptive sparse fcHMRF-LIS → "
    "FDR-controlled discovery maps → metrics and 3D visualization."
)

if run or "results" not in st.session_state:
    with st.spinner("Running the complete neuroimaging pipeline..."):
        start = time.perf_counter()

        g1, g2, truth, effect = generate_groups(
            int(seed),
            n1=8,
            n2=8,
            shape=(24, 24, 24),
            effect_size=float(effect_size),
            noise=float(noise),
        )

        _, p, z = voxel_statistics(g1, g2)

        roi_model = train_roi_network(42)
        X = build_features(z)
        roi_prob = roi_model.predict_proba(X)[:, 1].reshape(z.shape)
        roi_mask = roi_prob >= 0.50

        bh, bh_q = bh_mask(p, alpha)
        qmask, qvals = storey_qvalues(p, alpha)
        lfdr_mask, lfdr = local_fdr_mask(z, alpha)
        fc_mask, lis = adaptive_sparse_lis(
            z, roi_prob, alpha=alpha,
            iterations=iterations,
            smoothness=smoothness
        )

        elapsed = time.perf_counter() - start

        result_metrics = pd.DataFrame([
            {"Method": "Benjamini-Hochberg", **metrics(bh, truth)},
            {"Method": "q-value (Storey-style)", **metrics(qmask, truth)},
            {"Method": "LocalFDR (approx.)", **metrics(lfdr_mask, truth)},
            {"Method": "Adaptive sparse fcHMRF-LIS", **metrics(fc_mask, truth)},
        ])

        st.session_state.results = {
            "g1": g1, "g2": g2, "truth": truth, "effect": effect,
            "p": p, "z": z, "roi_prob": roi_prob, "roi_mask": roi_mask,
            "bh": bh, "bh_q": bh_q, "qmask": qmask, "qvals": qvals,
            "lfdr_mask": lfdr_mask, "lfdr": lfdr,
            "fc_mask": fc_mask, "lis": lis,
            "metrics": result_metrics, "runtime": elapsed,
        }

r = st.session_state.results

c1, c2, c3, c4 = st.columns(4)
c1.metric("3D voxels", f"{r['z'].size:,}")
c2.metric("Ground-truth ROI voxels", f"{r['truth'].sum():,}")
c3.metric("fcHMRF-LIS discoveries", f"{r['fc_mask'].sum():,}")
c4.metric("Runtime", f"{r['runtime']:.2f} s")

st.subheader("1. Experimental results")
st.dataframe(
    r["metrics"].style.format({
        "FDP": "{:.3f}",
        "FNP": "{:.3f}",
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "Dice": "{:.3f}",
    }),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    f"Target FDR α = {alpha:.2f}. For the synthetic experiment, ground-truth labels are known, "
    "so FDP/FNP/TP and Dice can be calculated exactly."
)

st.subheader("2. Deep-learning-based ROI detection")
roi1, roi2 = st.columns(2)
with roi1:
    st.metric("Neural ROI voxels", f"{r['roi_mask'].sum():,}")
    st.plotly_chart(
        surface_plot(r["roi_prob"], "Neural ROI probability", threshold=0.50),
        use_container_width=True,
    )
with roi2:
    st.metric("Mean ROI probability", f"{r['roi_prob'].mean():.3f}")
    st.plotly_chart(
        surface_plot(r["z"], "Voxel-wise z-statistics"),
        use_container_width=True,
    )

st.subheader("3. Adaptive sparse fcHMRF-LIS")
a, b = st.columns(2)
with a:
    st.metric("Mean LIS", f"{r['lis'].mean():.3f}")
    st.plotly_chart(
        surface_plot(1-r["lis"], "Estimated signal probability", threshold=1-alpha),
        use_container_width=True,
    )
with b:
    st.metric("LIS discoveries", f"{r['fc_mask'].sum():,}")
    st.plotly_chart(
        surface_plot(r["fc_mask"].astype(float), "fcHMRF-LIS discovery map", threshold=0.5),
        use_container_width=True,
    )

st.subheader("4. Ground truth and method comparison")
gt, cmp = st.columns(2)
with gt:
    st.plotly_chart(
        surface_plot(r["truth"].astype(float), "Known synthetic abnormality regions", threshold=0.5),
        use_container_width=True,
    )
with cmp:
    comparison = r["metrics"].set_index("Method")[["FDP", "FNP", "Dice", "Recall"]]
    st.bar_chart(comparison)

st.subheader("5. Download experiment results")
csv = r["metrics"].to_csv(index=False).encode("utf-8")
st.download_button(
    "Download metrics CSV",
    data=csv,
    file_name="fchmrf_lis_results.csv",
    mime="text/csv",
)

with st.expander("What this demo implements"):
    st.markdown(
        """
        **Implemented**
        - Synthetic 3D neuroimaging data with known abnormal regions.
        - Two-group voxel-wise statistical testing.
        - Lightweight neural-network ROI detection using a two-hidden-layer MLP.
        - Benjamini-Hochberg FDR baseline.
        - Storey-style q-value baseline.
        - Approximate LocalFDR baseline.
        - Adaptive sparse mean-field fcHMRF-LIS approximation.
        - LIS-based FDR-controlled rejection.
        - FDP, FNP, precision, recall, Dice, true positives and runtime.
        - Interactive 3D Plotly discovery maps.
        - CSV result export.

        **Important research limitation**
        This is an implementation-oriented prototype for demonstration. It uses sparse
        neighborhood filtering rather than reproducing the full paper's optimized
        fully-connected permutohedral-lattice implementation.
        """
    )

st.success(
    "Pipeline completed. This dashboard is suitable for demonstrating the proposed "
    "workflow and experimental design; it should not be presented as a clinical diagnostic tool."
)
