# 🧠 EEG Motor Imagery Decoding — From CSP to Deep Learning

![Banner](figs/day8_model_comparison.png)

> **Goal:** Build an end-to-end, reproducible pipeline for decoding imagined motor movements (left vs right hand) from EEG signals, benchmarking classical and deep-learning models.

---

## 📋 Overview

This project implements and benchmarks EEG decoding pipelines inspired by Brain–Computer Interface (BCI) research.
Using the **PhysioNet EEG Motor Movement/Imagery Dataset (EEGBCI)**, the workflow covers every stage of neural data processing — from raw signal preprocessing to cross-subject generalization with deep networks.

| Stage                 | Method                                                                         | Highlights                            |
| --------------------- | ------------------------------------------------------------------------------ | ------------------------------------- |
|  Preprocessing      | Filtering, Epoching, Artifact Rejection                                        | Built with **MNE-Python**             |
|  Feature Extraction | **Band-Power**, **Common Spatial Patterns (CSP)**, **Filterbank CSP (FB-CSP)** | Classical spatial filtering           |
|  Classification     | **LDA**, **EEGNet**, **ShallowFBCSPNet**                                       | From linear to deep models            |
|  Evaluation         | Group-wise CV, LOSO validation, κ / F1                                         | Reproducible and subject-independent  |
|  Interpretability   | Topographic maps, learned filters                                              | Links deep filters to neurophysiology |

---

## 📊 Dataset

**Dataset:** [PhysioNet EEG Motor Movement/Imagery Dataset (EEGBCI)](https://physionet.org/content/eegmmidb/1.0.0/)
**Subjects:** 109 healthy participants
**Sampling Rates:** 128 Hz or 160 Hz (depending on run version)
**Tasks:**

* Left vs Right Hand Motor Imagery
* Both Fists vs Both Feet (additional runs)
* Executed and Imagined Movements

Each run provides 64-channel EEG recordings following the **10–10 international montage**.

---

## 🧩 Project Structure

```bash
Motor-Imagery-EEG-Classifier/
├── data/                     # Raw EDF + processed FIF files
├── notebooks/
│   ├── 00_fetch_eeg_data.ipynb        # Download & organize EEGBCI dataset
│   ├── 01_explore_raw.ipynb           # Explore raw EEG & annotations
│   ├── 02_epoching.ipynb              # Filter, epoch & save per-run data
│   ├── 03_bandpower.ipynb             # Bandpower features & visual QC
│   ├── 04_csp_fb_csp.ipynb            # Classical pipelines
│   ├── 05_eegnet_baseline.ipynb       # EEGNet baseline
│   ├── 06_eegnet_tuning_crossval.ipynb# Tuning + GroupKFold/LOSO
│   └── 07_results_benchmark.ipynb     # Final results + figures
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   └── models.py
├── results/
│   ├── day7_loso.csv
│   ├── day8_final_summary.csv
│   └── ...
├── figs/
│   ├── csp_patterns.png
│   ├── day8_model_comparison.png
│   ├── day8_loso_bar.png
│   └── eegnet_kernel.png
├── environment.yml
├── Makefile
└── README.md
```

---

## ⚙️ Environment Setup

```bash
# Clone repo
git clone https://github.com/sumachinta/Motor-Imagery-EEG-Classifier.git
cd Motor-Imagery-EEG-Classifier

# Create environment
mamba env create -f environment.yml
conda activate neurodecode

# Launch notebooks
jupyter lab
```

---

## 🚀 Pipeline Summary

### 🧹 1. Preprocessing & Epoching

* **Band-pass filter:** 1–50 Hz
* **Notch:** 60 Hz + harmonics
* **Epoch window:** −0.2 s → 0.8 s around task cue
* **Baseline correction + artifact rejection**

```python
epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=0.8,
                    baseline=(None, 0), reject=dict(eeg=150e-6))
```

---

### 🔊 2. Feature Extraction

#### Bandpower features

* Alpha (8–13 Hz) & Beta (14–30 Hz) computed per channel
* Used for shallow classifiers (LDA / XGBoost)

#### CSP & FB-CSP

* Extracted spatial filters maximizing class variance difference
* Extended with filterbanks → 8–12, 12–16, 16–20… Hz bands

![CSP Patterns](figs/csp_patterns.png)
*Example CSP spatial patterns highlighting motor cortex regions (8–30 Hz)*

---

### 🧠 3. Deep Models

#### EEGNet

* Compact CNN designed for EEG decoding
* Depthwise + separable convolutions emulate spatial filters
* Trained with cross-subject GroupKFold validation

#### ShallowFBCSPNet

* Implements FBCSP-like spectral filtering in first conv layer
* Acts as bridge between CSP and CNNs

---

### ⚖️ 4. Evaluation & Generalization

| Split          | Description                   | Purpose               |
| -------------- | ----------------------------- | --------------------- |
| **GroupKFold** | 5 folds by subject            | Hyperparameter tuning |
| **LOSO**       | Leave-One-Subject-Out         | True generalization   |
| **Metrics**    | Accuracy, Cohen’s κ, macro F1 | Balanced evaluation   |

---

### 📈 5. Results

| Model           | Mean κ | ±SD  | Notes                    |
| --------------- | ------ | ---- | ------------------------ |
| FB-CSP + LDA    | 0.62   | 0.05 | Baseline spatial filter  |
| ShallowFBCSPNet | 0.65   | 0.04 | Deep-CSP equivalent      |
| EEGNet (tuned)  | 0.70   | 0.03 | Best cross-subject model |

![Benchmark](figs/day8_model_comparison.png)

**Takeaway:**
EEGNet achieved the best performance and generalization, while maintaining interpretable spatial-temporal filters resembling CSP maps.

---

### 🧩 6. Interpretability

| View           | Description                                               |
| -------------- | --------------------------------------------------------- |
| CSP topomaps   | spatial weight patterns focusing on C3/C4                 |
| EEGNet kernels | temporal filters highlighting μ (10 Hz) & β (20 Hz) bands |

![EEGNet Filters](figs/eegnet_kernel.png)

---

## 🧭 Key Learnings

* EEG decoding pipelines benefit from **supervised spatial filtering (CSP/FB-CSP)** for interpretability.
* **EEGNet** generalizes better across subjects when tuned via GroupKFold / LOSO.
* Cross-subject validation is essential to avoid **data leakage** and inflated metrics.
* CSP and EEGNet filters correspond to **sensorimotor rhythms**, showing physiological relevance.

---

## 🧰 Tech Stack

| Category       | Tools / Libraries                        |
| -------------- | ---------------------------------------- |
| EEG Processing | `MNE-Python`                             |
| ML / DL        | `Braindecode`, `PyTorch`, `scikit-learn` |
| Data           | `PhysioNet EEGBCI`                       |
| Visualization  | `matplotlib`, `seaborn`, `mne.viz`       |
| Environment    | `conda/mamba`, `Makefile`, `.env.yml`    |

---

## 📚 References

1. Ramoser H. *et al.* (2000). “Optimal spatial filtering of single trial EEG during imagined hand movement.” *IEEE Trans Rehabil Eng.*
2. Lawhern V. *et al.* (2018). “EEGNet: A Compact CNN for EEG-based BCIs.” *J Neural Eng.*
3. Schirrmeister R. *et al.* (2017). “Deep learning with convolutional networks for EEG decoding and visualization.” *Human Brain Mapping.*

---

## 🧩 Future Directions

* Extend to **4-class decoding** (Left, Right, Feet, Fists)
* Integrate **subject adaptation** (e.g., domain alignment)
* Explore **transformer-based architectures (EEG-ViT)**
* Real-time BCI prototype using **Muse/Emotiv** headset

---

## 👩‍💻 Author

**Suma Chinta**
Neural Data Scientist | CytoTronics | ex-Purdue Neuroscience
📍Boston, MA 🔗 [LinkedIn](https://linkedin.com/in/suma-chinta) 💻 [Portfolio](https://github.com/sumachinta)

---

## 🧠 Figures Gallery

<p align="center">
  <img src="figs/day8_model_comparison.png" width="45%"/>
  <img src="figs/day8_loso_bar.png" width="45%"/><br>
  <img src="figs/csp_patterns.png" width="45%"/>
  <img src="figs/eegnet_kernel.png" width="45%"/>
</p>

---

### ⭐ If you find this useful, give the project a star!
