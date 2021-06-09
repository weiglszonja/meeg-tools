import numpy as np
from mne import Epochs
from mne.preprocessing import bads, ICA
from autoreject import autoreject, Ransac
from random import sample

from utils.config import settings
from mne.utils import logger


def prepare_epochs_for_ica(epochs: Epochs) -> Epochs:
    """
    Drops epochs that were marked bad based on a global outlier detection.
    This implementation for the preliminary epoch rejection was based on the
    Python implementation of the FASTER algorithm from Marijn van Vliet
    https://gist.github.com/wmvanvliet/d883c3fe1402c7ced6fc
    Parameters
    ----------
    epochs

    Returns
    -------
    Epochs instance
    """
    logger.info('Preliminary epoch rejection: ')

    def _deviation(data: np.ndarray) -> np.ndarray:
        """
        Computes the deviation from mean for each channel.
        """
        channels_mean = np.mean(data, axis=2)
        return channels_mean - np.mean(channels_mean, axis=0)

    metrics = {
        'amplitude': lambda x: np.mean(np.ptp(x, axis=2), axis=1),
        'deviation': lambda x: np.mean(_deviation(x), axis=1),
        'variance': lambda x: np.mean(np.var(x, axis=2), axis=1),
    }

    epochs_data = epochs.get_data()

    bad_epochs = []
    for metric in metrics:
        scores = metrics[metric](epochs_data)
        outliers = bads._find_outliers(scores, threshold=3.0)
        logger.info(f'Bad epochs by {metric}\n\t{outliers}')
        bad_epochs.extend(outliers)

    bad_epochs = list(set(bad_epochs))
    epochs_faster = epochs.copy().drop(bad_epochs, reason='FASTER')

    return epochs_faster


def run_ica(epochs: Epochs) -> ICA:
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
    ica = ICA(n_components=settings['ica']['n_components'],
              random_state=42,
              method=settings['ica']['method'])
    ica.fit(epochs, decim=settings['ica']['decim'])

    if 'eog' not in epochs.get_channel_types():
        epochs.set_channel_types({'Fp1': 'eog', 'Fp2': 'eog'})

    eog_indices, eog_scores = ica.find_bads_eog(epochs)
    ica.exclude = eog_indices

    epochs.set_channel_types({'Fp1': 'eeg', 'Fp2': 'eeg'})

    return ica


def run_autoreject(epochs: Epochs, n_jobs: int = 11,
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


def run_ransac(epochs: Epochs, n_jobs: int = 11) -> Epochs:
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
