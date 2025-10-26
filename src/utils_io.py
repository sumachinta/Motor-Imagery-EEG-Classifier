# Reproducibility: set random seeds
import random
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score
from skorch import NeuralNetClassifier as EEGClassifier
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.dataset import ValidSplit
from pathlib import Path
from braindecode.models import EEGNet, ShallowFBCSPNet
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional


@dataclass
class EpochDataset:
    """Container for EEG epoch data and metadata"""

    X: np.ndarray #(n_train, n_ch, n_t)
    y: np.ndarray #(n_train,)
    sfreq : float #sampling frequency in Hz
    runs : Optional[np.ndarray] = None 
    subjects : Optional[np.ndarray] = None
    ch_names : Optional[list[str]] = None

    # Derived attributes
    @property
    def n_trials(self) -> int:
        return self.X.shape[0]
    @property
    def n_ch(self) -> int:
        return self.X.shape[1]
    @property
    def n_t(self) -> int:
        return self.X.shape[2]
    @property
    def n_classes(self) -> int:
        return int(np.unique(self.y).size)
    @property
    def classes_(self) -> np.ndarray:
        return np.unique(self.y)
    @property
    def info(self) -> str:
        info = (f"EpochDataset: {self.n_trials} trials, "
                f"{self.n_ch} channels, "
                f"{self.n_t} timepoints, "
                f"{self.n_classes} classes, "
                f"sfreq={self.sfreq} Hz")
        if self.subjects is not None:
            info += f", {np.unique(self.subjects).size} subjects"
        if self.runs is not None:
            info += f", {np.unique(self.runs).size} runs"
        return info

    
    


def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")        # Apple GPU
    elif torch.cuda.is_available():
        return torch.device("cuda")       # NVIDIA GPUs (not on Macs)
    else:
        return torch.device("cpu")
    

def make_eegnet(EpochData: EpochDataset, F1: int = 8, D: int = 2, drop: float = 0.25):
    return EEGNet(n_chans=EpochData.n_ch, 
                  n_outputs=EpochData.n_classes, 
                  n_times=EpochData.n_t, 
                  F1=F1, 
                  D=D, 
                  drop_prob=drop)


def eval_with_preproc(EpochData: EpochDataset, build_module, preproc_pair_fn=None, *, n_splits=5, plot_curves=False, saveFigs=False, filepath):
    """preproc_pair_fn(X_tr, X_te) -> (X_tr_prep, X_te_prep).
       If None, identity (no preprocessing)."""
    X = EpochData.X
    y = EpochData.y
    classes_ = EpochData.classes_
    groups = EpochData.subjects

    gkf = GroupKFold(n_splits=min(n_splits, np.unique(groups).size))
    baseline_acc = []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        Xtr, Xte = X[tr], X[te]
        if preproc_pair_fn is None:
            Xtr_p, Xte_p = Xtr, Xte
        else:
            Xtr_p, Xte_p = preproc_pair_fn(Xtr, Xte)

        clf = EEGClassifier(
                module=build_module(),
                criterion=nn.CrossEntropyLoss(),
                optimizer=torch.optim.Adam,
                lr=0.0005,
                batch_size=32,
                max_epochs=50,
                device=get_device(),
                train_split=ValidSplit(0.2, stratified=True, random_state=42),
                callbacks=[
                    ('es', EarlyStopping(patience=5, monitor='valid_loss')),
                    ('lr', LRScheduler('ReduceLROnPlateau', monitor='valid_loss', patience=5)),
                ], classes=classes_)
        
        clf.fit(Xtr_p, y[tr])
        if plot_curves:
            plot_training_curves(clf, "EEGNet baseline training")
        if saveFigs:
            save_training_curves(clf, filepath=filepath, fold=fold, label="EEGNet")
        yhat = clf.predict(Xte_p)
        baseline_acc.append({
            "fold": fold,
            "acc": accuracy_score(y[te], yhat),
            "kappa": cohen_kappa_score(y[te], yhat),
        })
    return baseline_acc

def summarize(baseline_acc, label):
    acc   = np.array([r["acc"] for r in baseline_acc])
    kappa = np.array([r["kappa"] for r in baseline_acc])
    print(f"{label:30s} acc {acc.mean():.3f}±{acc.std():.3f} | κ {kappa.mean():.3f}±{kappa.std():.3f}")


def plot_training_curves(clf, title="Training curves"):
    hist = clf.history
    plt.figure(figsize=(6,4))
    plt.plot(hist[:, 'train_loss'], label='Train loss')
    plt.plot(hist[:, 'valid_loss'], label='Valid loss')
    # if 'valid_accuracy' in hist[0]:
    #     plt.plot(hist[:, 'valid_accuracy'], label='Valid acc')
    plt.xlabel("Epoch")
    plt.legend()
    plt.title(title)
    plt.show()


def save_training_curves(clf: EEGClassifier, filepath: Path, fold = None, label = "EEGNet"):
    """
    Saves training & validation loss (and accuracy if available)
    to a PNG file instead of plotting inline.
    """

    hist = clf.history
    epochs = range(1, len(hist) + 1)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(epochs, hist[:, 'train_loss'], label='Train loss', color='tab:blue')
    ax.plot(epochs, hist[:, 'valid_loss'], label='Valid loss', color='tab:orange')
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    # if 'valid_accuracy' in hist[0]:
    #     ax2 = ax.twinx()
    #     ax2.plot(epochs, hist[:, 'valid_accuracy'], label='Valid acc', color='tab:green')
    #     ax2.set_ylabel("Accuracy")
    #     ax2.legend(loc='upper right')

    fname = f"{label}_fold{fold}_training_curves.png" if fold is not None else f"{label}_training_curves.png"
    filepath = Path(filepath) / fname
    plt.legend()
    plt.title(f"{label} Training curves")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close(fig)

    print(f"Training curves saved to {filepath}")


def zscore_per_trial_pair(Xtr, Xte, eps=1e-6):
# Best for EEGNet according to paper
# Best for within-subject or mixed-subject
    def _z(X):
        # mean/std over time axis for each (trial, channel)
        mu  = X.mean(axis=2, keepdims=True)
        sig = X.std(axis=2, keepdims=True)
        return ((X - mu) / (sig + eps)).astype(np.float32)
    return _z(Xtr), _z(Xte)


def foldwise_channel_standardize_pair(Xtr, Xte, eps=1e-6):
    # Best for cross-subject
    # compute per-channel mean/std on TRAIN fold across trials & time
    mu  = Xtr.mean(axis=(0, 2), keepdims=True)          # (1, C, 1)
    sig = Xtr.std(axis=(0, 2), keepdims=True)           # (1, C, 1)
    def _apply(X):
        return ((X - mu) / (sig + eps)).astype(np.float32)
    return _apply(Xtr), _apply(Xte)


# #momo