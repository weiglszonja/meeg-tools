import numpy as np
from mne import Epochs
from mne.preprocessing import bads, ICA
from autoreject import autoreject, Ransac
from random import sample

from .utils.config import settings
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

    If there are no EOG channels found, it tries to use 'Fp1' and 'Fp2' as EOG
    channels; if they are not found either, it chooses the first two channels
    to identify EOG components with mne.preprocessing.ica.find_bads_eog().
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
    ica_epochs = epochs.copy()
    ica.fit(ica_epochs, decim=settings['ica']['decim'])

    if 'eog' not in epochs.get_channel_types():
        if 'Fp1' and 'Fp2' in epochs.get_montage().ch_names:
            eog_channels = ['Fp1', 'Fp2']
        else:
            eog_channels = epochs.get_montage().ch_names[:2]
        logger.info('EOG channels are not found. Attempting to use '
                    f'{",".join(eog_channels)} channels as EOG channels.')
        ica_epochs.set_channel_types({ch: 'eog' for ch in eog_channels})

    eog_indices, _ = ica.find_bads_eog(ica_epochs)
    ica.exclude = eog_indices

    return ica


def run_autoreject(epochs: Epochs, n_jobs: int = 11,
                   subset: bool = False) -> autoreject.RejectLog:
    """
    Drop bad epochs based on AutoReject.
    Parameters
    ----------
    epochs: the instance to be cleaned
    n_jobs: the number of parallel processes to be run
    subset: whether to train autoreject on a random subset of data (faster)

    Returns
    -------
    Autoreject instance
    """
    ar = autoreject.AutoReject(random_state=42, n_jobs=n_jobs)

    n_epochs = len(epochs)
    if subset:
        logger.info(f'Fitting autoreject on random (n={int(n_epochs * 0.25)}) '
                    f'subset of epochs: ')
        subset = sample(set(np.arange(0, n_epochs, 1)), int(n_epochs * 0.25))
        ar.fit(epochs[subset])

    else:
        logger.info(f'Fitting autoreject on (n={n_epochs}) epochs: ')
        ar.fit(epochs)

    reject_log = ar.get_reject_log(epochs)

    return reject_log


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
            description=epochs_ransac.info[
                            'description'] + f', ({len(ransac.bad_chs_)}) '
                                             f'interpolated: ' + bads_str)
    else:
        epochs_ransac.info.update(
            description=epochs_ransac.info[
                            'description'] + ', (0) interpolated')

    return epochs_ransac
