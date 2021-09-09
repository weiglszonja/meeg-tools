import os
from pathlib import Path

import numpy as np
import pandas as pd
from mne import read_epochs
from meeg_tools.utils.epochs import create_new_events_from_metadata
from meeg_tools.time_frequency import compute_power

from tqdm import tqdm
from datetime import datetime

from mne.utils import logger

epochs_path = '/Volumes/crnl-memo-hd/preprocessed/epochs_asrt'
postfix = 'autoreject_ransac-epo.fif.gz'

event_ids = {'e1_L': 1, 'e2_L': 2, 'e3_L': 3, 'e4_L': 4, 'e5_L': 5,
             'e1_H': 6, 'e2_H': 7, 'e3_H': 8, 'e4_H': 9, 'e5_H': 10}


def run_tfr_pipeline(source: Path, target: Path):
    files = sorted(list(source.rglob(f'*{postfix}')))

    if not len(files):
        logger.warning(f"There are no files with the expected {postfix}."
                       f"Doing nothing ...")
        return

    if not target.exists():
        os.makedirs(target, exist_ok=False)

    pbar = tqdm(sorted(files))
    subjects_powers = []
    for file in pbar:
        pbar.set_description("Processing %s" % file.stem)
        fid = '_'.join(str(file).split('_')[:3])
        epochs = read_epochs(os.path.join(source, file), preload=False)

        if isinstance(epochs.metadata, pd.DataFrame):
            # make sure to analyze only correct answers
            epochs = epochs["answer == 'correct'"]
            # create separate event_ids for each condition (triplet H or L)
            # for each period (1 period = 5 blocks)
            epochs_tfr = create_new_events_from_metadata(epochs=epochs)
        else:
            epochs_tfr = epochs.copy()

        # make sure average ref is applied
        if not epochs_tfr.info['custom_ref_applied']:
            epochs_tfr.load_data().set_eeg_reference('average')

        epochs_tfr.info['fid'] = target / fid

        powers = compute_power(epochs=epochs_tfr, to_hdf5=False)
        subjects_powers.append(powers[np.newaxis, ...])

    pbar.close()

    subjects_powers_np = np.concatenate(subjects_powers, axis=0)
    fname = f'{datetime.utcnow().strftime("%Y%m%d%H%M%S")}_subjects_power.npy'
    np.save(target / fname, subjects_powers_np)


if __name__ == "__main__":
    epochs_in = Path(epochs_path)
    power_out = epochs_in.parent / 'power'
    run_tfr_pipeline(source=epochs_in, target=power_out)
