"""
This module contains functions that can be used to clean EEG/MEG data using MNE-Python.
https://github.com/weiglszonja/meeg-tools/blob/master/README.md
"""
from random import sample
import numpy as np

from mne import Epochs
from mne.io import RawArray
from mne.preprocessing import bads, ICA
from mne.utils import logger
from autoreject import autoreject
from pyprep import NoisyChannels

from .utils.config import settings


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
    logger.info("Preliminary epoch rejection: ")

    def _deviation(data: np.ndarray) -> np.ndarray:
        """
        Computes the deviation from mean for each channel.
        """
        channels_mean = np.mean(data, axis=2)
        return channels_mean - np.mean(channels_mean, axis=0)

    metrics = {
        "amplitude": lambda x: np.mean(np.ptp(x, axis=2), axis=1),
        "deviation": lambda x: np.mean(_deviation(x), axis=1),
        "variance": lambda x: np.mean(np.var(x, axis=2), axis=1),
    }

    epochs_data = epochs.get_data()

    bad_epochs = []
    for metric in metrics:
        scores = metrics[metric](epochs_data)
        outliers = bads._find_outliers(scores, threshold=3.0)
        logger.info(f"Bad epochs by {metric}\n\t{outliers}")
        bad_epochs.extend(outliers)

    bad_epochs = list(set(bad_epochs))
    epochs_faster = epochs.copy().drop(bad_epochs, reason="FASTER")

    return epochs_faster


def run_ica(epochs: Epochs, fit_params: dict = None) -> ICA:
    """
    Runs ICA decomposition on Epochs instance.

    If there are no EOG channels found, it tries to use 'Fp1' and 'Fp2' as EOG
    channels; if they are not found either, it chooses the first two channels
    to identify EOG components with mne.preprocessing.ica.find_bads_eog().
    Parameters
    ----------
    epochs: the instance to be used for ICA decomposition
    fit_params: parameters to be passed to ICA fit (e.g. orthogonal picard, extended infomax)
    Returns
    -------
    ICA instance
    """
    ica = ICA(
        n_components=settings["ica"]["n_components"],
        random_state=42,
        method=settings["ica"]["method"],
        fit_params=fit_params,
    )
    ica_epochs = epochs.copy()
    ica.fit(ica_epochs, decim=settings["ica"]["decim"])

    if "eog" not in epochs.get_channel_types():
        if "Fp1" and "Fp2" in epochs.get_montage().ch_names:
            eog_channels = ["Fp1", "Fp2"]
        else:
            eog_channels = epochs.get_montage().ch_names[:2]
        logger.info(
            "EOG channels are not found. Attempting to use "
            f'{",".join(eog_channels)} channels as EOG channels.'
        )
        ica_epochs.set_channel_types({ch: "eog" for ch in eog_channels})

    eog_indices, _ = ica.find_bads_eog(ica_epochs)
    ica.exclude = eog_indices

    return ica


def apply_ica(epochs: Epochs, ica: ICA) -> Epochs:
    """
    Applies ICA on Epochs instance.
    Parameters
    ----------
    epochs
    ica

    Returns
    -------

    """
    ica_epochs = epochs.copy().load_data()
    ica.apply(ica_epochs)

    ica_epochs.info.update(description=f"n_components: {len(ica.exclude)}")
    ica_epochs.info.update(temp=f'{epochs.info["temp"]}_ICA')

    return ica_epochs


def run_autoreject(
        epochs: Epochs, n_jobs: int = 11, subset: bool = False
) -> autoreject.RejectLog:
    """
    Finds bad epochs based on AutoReject.
    Parameters
    ----------
    epochs: the instance to be cleaned
    n_jobs: the number of parallel processes to be run
    subset: whether to train autoreject on a random subset of data (faster)

    Returns
    -------
    Autoreject instance
    """
    if not epochs.preload:
        epochs_autoreject = epochs.copy().load_data()
    else:
        epochs_autoreject = epochs.copy()

    ar = autoreject.AutoReject(random_state=42, n_jobs=n_jobs)

    n_epochs = len(epochs_autoreject)
    if subset:
        logger.info(
            f"Fitting autoreject on random (n={int(n_epochs * 0.25)}) "
            f"subset of epochs: "
        )
        subset = sample(set(np.arange(0, n_epochs, 1)), int(n_epochs * 0.25))
        ar.fit(epochs_autoreject[subset])

    else:
        logger.info(f"Fitting autoreject on (n={n_epochs}) epochs: ")
        ar.fit(epochs_autoreject)

    reject_log = ar.get_reject_log(epochs_autoreject)

    # report bad epochs where more than 15% (value updated from config.py)
    # of channels were marked as noisy within an epoch
    threshold = settings["autoreject"]["threshold"]
    num_bad_epochs = np.count_nonzero(reject_log.labels, axis=1)
    bad_epochs = np.where(num_bad_epochs > threshold * epochs.info["nchan"])
    bad_epochs = bad_epochs[0].tolist()

    reject_log.report = bad_epochs

    auto_bad_epochs = np.where(reject_log.bad_epochs)[0].tolist()
    if len(bad_epochs) < len(auto_bad_epochs):
        reject_log.report = sorted(
            list(set(bad_epochs + auto_bad_epochs)))

    logger.info(
        "\nAUTOREJECT report\n"
        f"There are {len(epochs_autoreject[reject_log.bad_epochs])} "
        f"bad epochs found with Autoreject. "
        f"You can assess these epochs with reject_log.bad_epochs\n"
        f"\nThere are {len(reject_log.report)} bad epochs where more than "
        f"{int(threshold * 100)}% of the channels were noisy. "
        f"You can assess these epochs with reject_log.report"
    )

    return reject_log


def apply_autoreject(epochs: Epochs, reject_log: autoreject.RejectLog) -> Epochs:
    """
    Drop bad epochs based on AutoReject.
    Parameters
    ----------
    epochs
    reject_log

    Returns
    -------

    """
    epochs_autoreject = epochs.copy().drop(reject_log.report, reason="AUTOREJECT")
    epochs_autoreject.info.update(temp=f'{epochs.info["temp"]}_autoreject')

    return epochs_autoreject


def get_noisy_channels(epochs: Epochs, with_ransac: bool = False) -> list:
    """
    Find bad channels using a range of methods as described in the PREP pipeline.
    Note that low-frequency trends should be removed from the EEG signal prior
    to bad channel detection.
    Read the documentation for further information about the methods:
    https://pyprep.readthedocs.io/en/latest/generated/pyprep.NoisyChannels.html#pyprep.NoisyChannels

    References
    ----------
    Bigdely-Shamlo, N., Mullen, T., Kothe, C., Su, K. M., Robbins, K. A. (2015).
    The PREP pipeline: standardized preprocessing for large-scale EEG analysis.
    Frontiers in Neuroinformatics, 9, 16.

    Parameters
    ----------
    epochs: Epochs object to use for bad channels detection
    with_ransac: whether RANSAC should be used for bad channel detection,
    in addition to the other methods.

    Returns
    -------
    list of bad channels names detected
    """
    # transform epochs to continuous data
    # to shape of (n_channels, n_epochs, n_times)
    data = np.transpose(epochs.get_data(), (1, 0, 2))
    # reshape to (n_channels, n_epochs * n_times) continuous data
    data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))
    # create Raw object from continuous data
    raw = RawArray(data=data, info=epochs.info)

    noisy_channels = NoisyChannels(raw=raw, do_detrend=False, random_state=42)
    noisy_channels.find_all_bads(ransac=with_ransac)

    bads = noisy_channels.get_bads(verbose=False)
    if bads:
        logger.info(
            "\nNoisyChannels REPORT\n"
            "------------------------"
            f"\n{np.round(len(bads) / len(epochs.ch_names), 2) * 100}%"
            f" of the channels were detected as noisy."
            f'\n({len(bads)}) channels: {", ".join(bads)}'
        )
    return bads


def interpolate_bad_channels(epochs: Epochs, bads: list) -> Epochs:
    """
    Interpolates channels in an Epochs instance based on a list of channel names.
    Parameters
    ----------
    epochs
    bads

    Returns
    -------

    """
    if not epochs.preload:
        epochs_interpolated = epochs.copy().load_data()
    else:
        epochs_interpolated = epochs.copy()

    epochs_interpolated.info["bads"] = bads

    if bads:
        bads_str = ", ".join(bads)
        description = f", interpolated: {bads_str}"
        epochs_interpolated.info.update(
            description=epochs.info["description"] + description
        )

    epochs_interpolated.interpolate_bads(reset_bads=True)
    epochs_interpolated.info.update(temp=f'{epochs.info["temp"]}_ransac')
    return epochs_interpolated
