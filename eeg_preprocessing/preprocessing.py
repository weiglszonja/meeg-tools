import mne
from autoreject import autoreject, Ransac

from .utils.faster import faster_bad_epochs
import numpy as np
from random import sample


def prepare_epochs_for_ica(epochs: mne.Epochs) -> mne.Epochs:
    """
    Prepares epochs for ICA:
    Drops epochs that were marked bad based on outlier detection.
    Parameters
    ----------
    epochs

    Returns
    -------
    Epochs instance
    """
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

    epochs.set_channel_types({'Fp1': 'eeg', 'Fp2': 'eeg'})

    return ica


def run_autoreject(epochs: mne.Epochs, n_jobs: int = 11,
                   subset: bool = False) -> autoreject.AutoReject:
    """
    Drop bad epochs based on AutoReject.
    Parameters
    ----------
    epochs: the instance to be cleaned
    n_jobs: the number of parallel processes to be run
    subset: whether to train autoreject on a random subset of data (faster) default is False.

    Returns
    -------
    Autoreject instance
    """
    ar = autoreject.AutoReject(random_state=42, n_jobs=n_jobs)

    n_epochs = len(epochs)
    if subset:

        print(f'Fitting autoreject on random (n={int(n_epochs * 0.25)}) subset of epochs: ')
        subset = sample(set(np.arange(0, n_epochs, 1)), int(n_epochs * 0.25))
        ar.fit(epochs[subset])

    else:
        print(f'Fitting autoreject on (n={n_epochs}) epochs: ')
        ar.fit(epochs)

    return ar


def run_ransac(epochs: mne.Epochs, n_jobs: int = 11) -> mne.Epochs:
    """
    Find and interpolate bad channels with Ransac.
    If there are no bad channels found returns the Epochs instance unmodified.
    Parameters
    ----------
    epochs: the instance where bad channels to be found
    n_jobs: the number of parallel processes to run

    Returns
    -------
    Epochs instance
    """
    ransac = Ransac(verbose='progressbar', n_jobs=n_jobs)
    epochs_ransac = ransac.fit_transform(epochs)
    if ransac.bad_chs_:
        bads_str = ', '.join(ransac.bad_chs_)
        epochs_ransac.info.update(
            description=epochs_ransac.info['description'] + ', interpolated: ' + bads_str)

    return epochs_ransac
