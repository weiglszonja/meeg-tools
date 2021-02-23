"""
Author: Marijn van Vliet
Forked from: https://gist.github.com/wmvanvliet/d883c3fe1402c7ced6fc
"""
import numpy as np
import mne
from mne.preprocessing import find_outliers
from mne.utils import logger


def _deviation(data):
    """Computes the deviation from mean for each channel in a set of epochs.
    This is not implemented as a lambda function, because the channel means
    should be cached during the computation.

    Parameters
    ----------
    data : 3D numpy array
        The epochs (#epochs x #channels x #samples).
    Returns
    -------
    dev : 1D numpy array
        For each epoch, the mean deviation of the channels.
    """
    ch_mean = np.mean(data, axis=2)
    return ch_mean - np.mean(ch_mean, axis=0)


def faster_bad_epochs(epochs, picks=None, thres=3, use_metrics=None):
    """Implements the second step of the FASTER algorithm.

    This function attempts to automatically mark bad epochs by performing
    outlier detection.
    Parameters
    ----------
    epochs : Instance of Epochs
        The epochs to analyze.
    picks : list of int | None
        Channels to operate on. Defaults to EEG channels.
    thres : float
        The threshold value, in standard deviations, to apply. An epoch
        crossing this threshold value is marked as bad. Defaults to 3.
    use_metrics : list of str
        List of metrics to use. Can be any combination of:
            'amplitude', 'variance', 'deviation'
        Defaults to all of them.
    Returns
    -------
    bads : list of int
        The indices of the bad epochs.
    """

    metrics = {
        'amplitude': lambda x: np.mean(np.ptp(x, axis=2), axis=1),
        'deviation': lambda x: np.mean(_deviation(x), axis=1),
        'variance': lambda x: np.mean(np.var(x, axis=2), axis=1),
    }

    if picks is None:
        picks = mne.pick_types(epochs.info, meg=False, eeg=True,
                               exclude='bads')
    if use_metrics is None:
        use_metrics = metrics.keys()

    data = epochs.get_data()[:, picks, :]

    bads = []
    for m in use_metrics:
        s = metrics[m](data)
        b = find_outliers(s, thres)
        logger.info('Bad by %s:\n\t%s' % (m, b))
        bads.append(b)

    return np.unique(np.concatenate(bads)).tolist()