from typing import List, Tuple
from pathlib import Path
import mne
from mne.datasets import eegbci

def load_eegbci_raws(subjects: List[int], runs: List[int], cache_dir: str | Path | None = None
                    ) -> Tuple[List[mne.io.BaseRaw], List[str]]:
    """
    Auto-downloads EEGBCI EDFs (if missing) and returns Raw objects and paths.
    """
    path = None if cache_dir is None else str(Path(cache_dir))
    files = eegbci.load_data(subjects=subjects, runs=runs, path=path, update_path=True)
    raws = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in files]
    return raws, files
