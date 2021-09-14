import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from mne import read_epochs
from meeg_tools.utils.epochs import create_new_events_from_metadata
from meeg_tools.time_frequency import compute_power

from tqdm import tqdm
from mne.utils import logger

# file path to preprocessed epochs
epochs_path = '/Volumes/crnl-memo-hd/preprocessed/epochs_asrt'
# postfix to lookup when collecting the files
postfix = 'autoreject_ransac-epo.fif.gz'


def run_tfr_pipeline(source: Path, target: Path):
    files = sorted(list(source.rglob(f'*{postfix}')))

    if not len(files):
        logger.warning(f"There are no files with the expected {postfix}."
                       f"Doing nothing ...")
        return

    pbar = tqdm(sorted(files))
    subjects_powers = []
    for file in pbar:
        pbar.set_description("Processing %s" % file.stem)
        fid = '_'.join(str(file.stem.replace(' ', '_')).split('_')[:3])

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

        powers = compute_power(epochs=epochs_tfr, to_hdf5=True)
        subjects_powers.append(powers[np.newaxis, ...])

    pbar.close()

    subjects_powers_np = np.concatenate(subjects_powers, axis=0)
    fname = f'{datetime.utcnow().strftime("%Y%m%d%H%M%S")}_subjects_power.npy'
    np.save(target / fname, subjects_powers_np)


if __name__ == "__main__":
    epochs_in = Path(epochs_path)
    power_out = epochs_in.parent / 'power'
    run_tfr_pipeline(source=epochs_in, target=power_out)
