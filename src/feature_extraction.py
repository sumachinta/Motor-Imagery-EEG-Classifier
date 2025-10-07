import numpy as np
import mne

BANDS = {"alpha": (8,13), "beta": (13,30)} 
# BANDS = {"alpha": (8,12), "beta": (12,16), "gamma": (16,26), "high_gamma": (26,30)}

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
