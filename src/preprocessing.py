from typing import Dict, Tuple, Optional, List
import numpy as np
import mne
from pathlib import Path
import re

def notch_powerline(raw: mne.io.BaseRaw, line_freq: int = 60):
    sfreq = raw.info['sfreq']
    nyQ = sfreq / 2
    # builds a list of frequencies to notch filter < Nyquist
    freqs = np.arange(line_freq, nyQ, line_freq)
    return raw.notch_filter(freqs, verbose=False)



def raw_data_filter(raw: mne.io.BaseRaw, line_freq: int = 60) -> mne.io.BaseRaw:
    """Notch, bandpass (1–40 Hz), average reference."""
    raw = raw.copy().load_data()
    raw = notch_powerline(raw, line_freq)
    raw.filter(l_freq=1., h_freq=40., phase='zero', fir_design='firwin', verbose=False)
    raw, _ = mne.set_eeg_reference(raw, 'average')
    return raw

# ---------- EEGBCI-specific events ----------
# Per PhysioNet:
# Runs 3,7,11: Task 1 (real fists L/R) ; 4,8,12: Task 2 (imagined fists L/R)
# Runs 5,9,13: Task 3 (real both-fists vs both-feet) ; 6,10,14: Task 4 (imagined both-fists vs both-feet)
# T0=rest; T1=left (or both-fists); T2=right (or both-feet)
MI_map = {"T0": "Rest", "T1": "Left MI", "T2": "Right MI"}

_EEGBCI_FIST_RUNS = {3, 4, 7, 8, 11, 12}
_EEGBCI_BOTH_RUNS = {5, 6, 9, 10, 13, 14}

def _infer_subject_number_from_fname(fname: str | Path) -> Optional[int]:
    subj_name = re.match(r'(?i)S0*(\d+)R0*(\d+)\.edf$', fname)
    subj_number = int(subj_name.group(1)) if subj_name else None
    return subj_number

def _infer_run_number_from_fname(fname: str | Path) -> Optional[int]:
    m = re.search(r"R(\d{2})\.edf$", str(fname))
    return int(m.group(1)) if m else None



def get_events_and_ids_eegbci(raw: mne.io.BaseRaw) -> Tuple[np.ndarray, Dict[str,int], Dict[int,str], int]:
    """Returns (events, event_id, inv_map, run).
    events: (n_events, 3) array of (onset, 0, event_id)
    event_id: mapping of event name to id
    inv_map: mapping of id to event name
    run: inferred run number from filename (or -1 if unknown)
    Note: recodes event ids to global scheme: left(1), right(2), both_fists(3), both_feet(4) based on run type. Ignores rest(T0) events.
      """
    fname = (raw.filenames[0] if getattr(raw, "filenames", None) else "") or ""
    run = _infer_run_number_from_fname(fname) or -1

    events, anno_map = mne.events_from_annotations(raw, verbose=False)
    code_T1, code_T2 = anno_map.get('T1'), anno_map.get('T2')

    if code_T1 is None and code_T2 is None:
        return events[:0], {"left":1,"right":2,"both_fists":3,"both_feet":4}, {1:"left",2:"right",3:"both_fists",4:"both_feet"}, run
    
    # keep only T1/T2 events (drop T0/rest)
    keep = np.isin(events[:, 2], [c for c in (code_T1, code_T2) if c is not None])
    events = events[keep].copy()

    # recode to global ids based on run type
    if run in _EEGBCI_BOTH_RUNS:
        # T1 → both_fists(3), T2 → both_feet(4)
        code_map = {code_T1: 2, code_T2: 3}
    else:
        # default & fist runs: T1 → left(1), T2 → right(2)
        code_map = {code_T1: 0, code_T2: 1}

    events[:, 2] = np.array([code_map.get(c, c) for c in events[:, 2]], dtype=int)
    event_id_full = {"left":0, "right":1, "both_fists":2, "both_feet":3}
    # keep only event_ids for present run
    event_id = {k:v for k,v in event_id_full.items() if v in set(np.unique(events[:,2]))}
    inv_map  = {v:k for k,v in event_id.items()}
    events[:, 2] = events[:, 2].astype(int)
    return events, event_id, inv_map, run



