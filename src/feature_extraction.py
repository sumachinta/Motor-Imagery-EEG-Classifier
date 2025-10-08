import numpy as np
import mne

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.random_projection import SparseRandomProjection
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding as LLE
# import umap  # pip install umap-learn

# BANDS = {"alpha": (8,13), "beta": (13,30)} 
BANDS = {"alpha": (8,12), "beta": (12,16), "gamma": (16,26), "high_gamma": (26,30)}

def epochs_to_bandpower(epochs, bands=BANDS, log=True):
    """Return X (n_epochs, n_chan* n_bands), y (labels), ch_names, band_names."""
    # PSD over the whole epoch; you can also use time-frequency later
    psd = epochs.compute_psd(fmin=1, fmax=40, method="welch", picks="eeg")
    psd_data, freqs = psd.get_data(return_freqs=True)   # (n_epochs, n_ch, n_freq)
    bp_list, band_names = [], []
    for name, (lo, hi) in bands.items():
        idx = np.logical_and(freqs>=lo, freqs<=hi)
        bp = psd_data[:, :, idx].mean(axis=-1)          # mean power in band
        if log:
            bp = np.log10(bp + 1e-12)                   # stabilize
        bp_list.append(bp)                               # (n_ep, n_ch)
        band_names.append(name)
    X = np.concatenate(bp_list, axis=1)                  # (n_ep, n_ch * n_bands)
    y = epochs.events[:, -1]
    chs = epochs.ch_names
    return X, y, chs, band_names






def reduce_features(X, y=None, method="pca", n_components=2, standardize=True, random_state=42):
    """
    Dimensionality reduction for EEG feature matrices.
    X: (n_samples, n_features) e.g., (trials, 128)
    y: labels (needed for 'lda')
    method: 'pca' | 'svd' | 'ica' | 'lda' | 'rp' | 'tsne' | 'kpca' | 'isomap' | 'lle' | 'umap'
    Returns: Z (n_samples, n_components), fitted_model, info(dict)
    """
    X_in = X.copy()
    scaler = None
    if standardize and method not in ("rp",):  # RP works fine without scaling, others benefit
        scaler = StandardScaler()
        X_in = scaler.fit_transform(X_in)

    info = {}
    if method == "pca":
        model = PCA(n_components=n_components, random_state=random_state)
        Z = model.fit_transform(X_in)
        info["expl_var"] = model.explained_variance_ratio_
    elif method == "svd":   # linear, good for high-dim; like PCA on centered data
        model = TruncatedSVD(n_components=n_components, random_state=random_state)
        Z = model.fit_transform(X_in)
        info["expl_var"] = model.explained_variance_ratio_
    elif method == "ica":
        model = FastICA(n_components=n_components, random_state=random_state, max_iter=1000)
        Z = model.fit_transform(X_in)
    elif method == "lda":
        if y is None:
            raise ValueError("LDA requires y labels.")
        # n_components must be <= (n_classes-1)
        n_classes = np.unique(y).size
        n_components = min(n_components, n_classes - 1)
        model = LDA(n_components=n_components)
        Z = model.fit(X_in, y).transform(X_in)
    elif method == "rp":
        model = SparseRandomProjection(n_components=n_components, random_state=random_state)
        Z = model.fit_transform(X_in)
    elif method == "tsne":
        model = TSNE(n_components=n_components, init="pca", learning_rate="auto",
                     perplexity=min(30, max(5, X.shape[0]//10)), random_state=random_state)
        Z = model.fit_transform(X_in)
    elif method == "kpca":
        model = KernelPCA(n_components=n_components, kernel="rbf", gamma=1/X_in.shape[1], random_state=random_state)
        Z = model.fit_transform(X_in)
    elif method == "isomap":
        model = Isomap(n_neighbors=10, n_components=n_components)
        Z = model.fit_transform(X_in)
    elif method == "lle":
        model = LLE(n_neighbors=10, n_components=n_components, method="standard")
        Z = model.fit_transform(X_in)
    # elif method == "umap":
    #     model = umap.UMAP(n_components=n_components, n_neighbors=15, min_dist=0.1, random_state=random_state)
    #     Z = model.fit_transform(X_in)
    else:
        raise ValueError(f"Unknown method: {method}")

    return Z, model, info

