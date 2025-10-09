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
_EEGBCI_FIST_RUNS = {3, 4, 7, 8, 11, 12}
_EEGBCI_BOTH_RUNS = {5, 6, 9, 10, 13, 14}

def _infer_run_number_from_fname(fname: str | Path) -> Optional[int]:
    m = re.search(r"R(\d{2})\.edf$", str(fname))
    return int(m.group(1)) if m else None

# def eegbci_event_map_for_run(run: int, two_class_only: bool = True) -> Dict[str, int]:
#     """
#     Build EEGBCI event_id mapping for this run.
#     If two_class_only=True:
#        - fist runs -> {'left': 1, 'right': 2}
#        - both runs -> {'both_fists': 1, 'both_feet': 2}
#       (rest T0 is ignored for classification)
#     """
#     if run in _EEGBCI_FIST_RUNS:
#         return {"left": 1, "right": 2} if two_class_only else {"rest": 0, "left": 1, "right": 2}
#     if run in _EEGBCI_BOTH_RUNS:
#         return {"both_fists": 1, "both_feet": 2} if two_class_only else {"rest": 0, "both_fists": 1, "both_feet": 2}
#     # default: treat like fists (safe fallback)
#     return {"left": 1, "right": 2}


# def get_events_and_ids_eegbci(raw: mne.io.BaseRaw) -> Tuple[np.ndarray, Dict[str,int], Dict[int,str], int]:
#     """Returns (events, event_id, inv_map, run).
#     events: (n_events, 3) array of (onset, 0, event_id)
#     event_id: mapping of event name to id
#     inv_map: mapping of id to event name
#     run: inferred run number from filename (or -1 if unknown)
#     Note: recodes event ids to global scheme: left(1), right(2), both_fists(3), both_feet(4) based on run type. Ignores rest(T0) events.
#       """
#     fname = (raw.filenames[0] if getattr(raw, "filenames", None) else "") or ""
#     run = _infer_run_number_from_fname(fname) or -1

#     events, anno_map = mne.events_from_annotations(raw, verbose=False)
#     code_T1, code_T2 = anno_map.get('T1'), anno_map.get('T2')

#     if code_T1 is None and code_T2 is None:
#         return events[:0], {"left":1,"right":2,"both_fists":3,"both_feet":4}, {1:"left",2:"right",3:"both_fists",4:"both_feet"}, run
    
#     # keep only T1/T2 events (drop T0/rest)
#     keep = np.isin(events[:, 2], [c for c in (code_T1, code_T2) if c is not None])
#     events = events[keep].copy()

#     # recode to global ids based on run type
#     if run in _EEGBCI_BOTH_RUNS:
#         # T1 → both_fists(3), T2 → both_feet(4)
#         code_map = {code_T1: 3, code_T2: 4}
#     else:
#         # default & fist runs: T1 → left(1), T2 → right(2)
#         code_map = {code_T1: 1, code_T2: 2}

#     events[:, 2] = np.array([code_map.get(c, c) for c in events[:, 2]], dtype=int)
#     event_id = {"left":1, "right":2, "both_fists":3, "both_feet":4}
#     inv_map  = {v:k for k,v in event_id.items()}
#     events[:, 2] = events[:, 2].astype(int)
#     return events, event_id, inv_map, run


def get_events_and_ids_eegbci(raw: mne.io.BaseRaw, two_class_only: bool = True ) -> Tuple[np.ndarray, Dict[str,int], Dict[int,str], int]: 
    """Return (events, event_id, inv_map, run) with events[:,2] recoded to match event_id.""" 
    fname = (raw.filenames[0] if getattr(raw, "filenames", None) else "") or "" 
    run = _infer_run_number_from_fname(fname) or -1 
    events, anno_map = mne.events_from_annotations(raw, verbose=False) 
    # Original ints for T0/T1/T2 (order can vary; read from anno_map) 
    code_T0 = anno_map.get('T0', None) 
    code_T1 = anno_map.get('T1', None) 
    code_T2 = anno_map.get('T2', None) 
    
    if run in _EEGBCI_BOTH_RUNS: 
        # Both-fists vs both-feet runs 
        desired_names = ('both_fists', 'both_feet') 
    else: 
        # Fist left vs right runs (default) 
        desired_names = ('left', 'right') 
        if two_class_only: 
            event_id = {desired_names[0]: 1, desired_names[1]: 2} 
        else: 
            event_id = {'rest': 0, desired_names[0]: 1, desired_names[1]: 2} 
            # Filter to the two classes (drop rest) 
            keep_codes = [c for c in (code_T1, code_T2) if c is not None] 
            mask = np.isin(events[:, 2], keep_codes) 
            events = events[mask] 
            # Recode: T1 -> 1, T2 -> 2 (match event_id) 
            recoded = events[:, 2].copy() 
            if code_T1 is not None: 
                recoded[recoded == code_T1] = 1 
            if code_T2 is not None: 
                recoded[recoded == code_T2] = 2 

            events[:, 2] = recoded.astype(int) 
            inv_map = {v: k for k, v in event_id.items()} 
            return events, event_id, inv_map, run


