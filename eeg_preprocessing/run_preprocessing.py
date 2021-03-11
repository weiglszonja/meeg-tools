import os
import mne
from autoreject import autoreject, Ransac

from .utils.faster import faster_bad_epochs
import numpy as np
import pandas as pd
from random import sample


def create_epochs_for_ica(raw: mne.io.Raw) -> mne.Epochs:
    """
    Creates fixed length continuous epochs for ICA.
    Drops epochs that were marked bad based on outlier detection.
    Parameters
    ----------
    raw: the instance to be used for creating the epochs

    Returns
    -------
    Epochs instance
    """
    # remove slow drifts and high freq noise
    raw_bandpass = raw.load_data().copy().filter(l_freq=0.5, h_freq=45)

    # create fixed length epoch for preprocessing
    epoch_duration_in_seconds = 1.0
    events = mne.make_fixed_length_events(raw_bandpass,
                                          id=1,
                                          first_samp=True,
                                          duration=epoch_duration_in_seconds)

    epochs = mne.Epochs(raw=raw_bandpass,
                        events=events,
                        picks='eeg',
                        event_id=1,
                        baseline=None,
                        tmin=0.,
                        tmax=epoch_duration_in_seconds - (1 / raw.info['sfreq']),
                        preload=True)

    # remove from memory
    del raw_bandpass

    print('Preliminary epoch rejection: ')
    bad_epochs = faster_bad_epochs(epochs, picks=None, thres=3, use_metrics=None)
    epochs_faster = epochs.copy().drop(bad_epochs, reason='FASTER')

    return epochs_faster


def run_ica(epochs: mne.Epochs) -> mne.preprocessing.ica:
    """
    Runs ICA decomposition on Epochs instance.

    If there are no EOG channels found, it uses 'Fp1' and 'Fp2' channels to identify
    and mark EOG components.
    Parameters
    ----------
    epochs: the instance to be used for ICA decomposition

    Returns
    -------
    ICA instance
    """
    ica = mne.preprocessing.ICA(n_components=32, random_state=42, method='infomax')
    ica.fit(epochs, decim=2)

    if 'eog' not in epochs.get_channel_types():
        epochs.set_channel_types({'Fp1': 'eog', 'Fp2': 'eog'})

    eog_indices, eog_scores = ica.find_bads_eog(epochs)
    ica.exclude = eog_indices

    ica.plot_sources(epochs, start=0, stop=10)

    epochs.set_channel_types({'Fp1': 'eeg', 'Fp2': 'eeg'})

    return ica


def run_autoreject(epochs: mne.Epochs, n_jobs: int = 11, subset: bool = False) -> mne.Epochs:
    """
    Drop bad epochs based on AutoReject.
    Parameters
    ----------
    epochs: the instance to be cleaned
    n_jobs: the number of parallel processes to be run
    subset: whether to train autoreject on a random subset of data (faster) default is False.

    Returns
    -------
    Epochs instance
    """
    ar = autoreject.AutoReject(random_state=42, n_jobs=n_jobs)

    if subset:
        n_epochs = len(epochs)
        print(f'Fitting autoreject on random (n={n_epochs} subset of epochs: ')
        subset = sample(set(np.arange(0, n_epochs, 1)), int(n_epochs * 0.25))
        ar.fit(epochs[subset])

    else:
        print('Fitting autoreject on epochs: ')
        ar.fit(epochs)

    reject_log = ar.get_reject_log(epochs)
    epochs.drop(reject_log.bad_epochs, reason='AUTOREJECT')
    epochs.info['reject_log'] = reject_log

    return epochs


def run_ransac(epochs: mne.Epochs, n_jobs: int = 11) -> mne.Epochs:
    # find bad channels with ransac
    ransac = Ransac(verbose='progressbar', n_jobs=n_jobs)
    epochs_ransac = ransac.fit_transform(epochs)
    if ransac.bad_chs_:
        bads_str = ', '.join(ransac.bad_chs_)
        epochs_ransac.info.update(
            description=epochs_ransac.info['description'] + ' interpolated: ' + bads_str)

    return epochs_ransac

#
# if __name__ == '__main__':
#     # Set base path to EEG data
#     base_path = 'G:/TMS_rewiring/'
#     eeg_path = os.path.join(base_path, 'Raw_data/24_L/Day1/EEG/')
#     # Create folder for preprocessed and interim files
#     folder_name = 'preprocessed'
#     interim_path = os.path.join(base_path, folder_name)
#
#     raw_file_name = [file for file in os.listdir(eeg_path) if file.endswith('.vhdr')][0]
#     subject, condition, day = raw_file_name.split('_')
#     num_day = [d for d in day if d.isdigit()][0]
#
#     # read raw file (not loaded into memory)
#     raw = mne.io.read_raw_brainvision(os.path.join(eeg_path, raw_file_name), preload=False, verbose=True)
#
#     resting_events, block_events = get_events_from_raw(raw)
#
#     # Create path to interim epoch files
#     interim_epochs_path = os.path.join(interim_path, condition, 'epochs_rs')
#     if not os.path.exists(interim_epochs_path):
#         os.makedirs(interim_epochs_path)
#
#     # Create path to interim raw files
#     interim_raw_path = os.path.join(interim_path, condition, 'raw_rs')
#     if not os.path.exists(interim_raw_path):
#         os.makedirs(interim_raw_path)
#
#     raw_rs = concat_raws_from_events(resting_events)
#     epochs_ica = run_ica(raw=raw_rs)
#
#