"""
This module contains functions that can be used to perform time-frequency analysis on
EEG/MEG data with MNE-Python.
"""
import os
import re

import numpy as np

from mne import Epochs
from mne.time_frequency import tfr_morlet, AverageTFR
from mne.utils import logger
from yasa import irasa


def compute_power(epochs: Epochs, config: dict) -> AverageTFR:
    """
    Computes Time-Frequency Representation (TFR) using complex Morlet wavelets
    averaged over epochs. Power is written to an HDF5 file.
    Parameters
    ----------
    epochs
    config
    """
    fmin = config["morlet"]["fmin"]
    fmax = config["morlet"]["fmax"]
    step = config["morlet"]["step"]
    freqs = np.logspace(*np.log10([fmin, fmax]), num=step)
    n_cycles = freqs / 2.0

    power = tfr_morlet(
        epochs.average() if config["is_evoked"] else epochs.copy(),
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=False,
        return_itc=False,
        decim=config["morlet"]["decim"],
        average=True,
        n_jobs=-1,
    )

    return power


def save_to_hdf5(power: AverageTFR):
    if not os.path.exists(power.comment.parent):
        os.makedirs(power.comment.parent)

    power.comment = str(power.comment)
    # replace floats in file name with integers
    floats = re.findall("[-+]?\d*\.\d+", power.comment)
    if floats:
        for num in floats:
            power.comment = power.comment.replace(num, str(int(float(num))))
    file_path_with_extension = f"{power.comment}_power-tfr.h5"

    logger.info(f"Saving power at {file_path_with_extension} ...")
    power.save(fname=str(file_path_with_extension), overwrite=True)
    logger.info("[FINISHED]")


def average_power_into_frequency_bands(power: AverageTFR, config: dict) -> AverageTFR:
    """
    Computes average power in specific frequency ranges defined in config
    Parameters
    ----------
    power
    config

    Returns
    -------

    """
    band_values = config.values()
    band_power_data_arr = []

    for band in band_values:
        band_filter = np.logical_and(
            power.freqs >= float(band[0]), power.freqs < float(band[1])
        )
        if int(max(power.freqs)) == int(max(band_values)[1]):
            band_filter[-1] = True
        band_data = power.data[:, band_filter, :].mean(axis=1)
        band_power_data_arr.append(band_data[np.newaxis, :])

    band_power_data = np.concatenate(band_power_data_arr, axis=0)
    band_power_data = np.transpose(band_power_data, (1, 0, 2))

    band_freqs = np.asarray([band[0] for band in band_values])

    band_power = power.copy()
    band_power.data = band_power_data
    band_power.freqs = band_freqs
    band_power.comment = power.comment + "_band_average"

    return band_power


def compute_power_irasa_method(epochs: Epochs, bands: tuple):
    """
    Computes power by separating the aperiodic (= fractal, or 1/f) and oscillatory
    component of the power spectra of EEG data using the IRASA method.
    https://raphaelvallat.com/yasa/build/html/generated/yasa.irasa.html
    Parameters
    ----------
    epochs
    bands

    Returns
    -------

    """
    # transform epochs to continuous data
    # to shape of (n_channels, n_epochs, n_times)
    data = np.transpose(epochs.get_data(), (1, 0, 2))
    # reshape to (n_channels, n_epochs * n_times) continuous data
    data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))

    freqs, _, psd_oscillatory, _ = irasa(
        data=data,
        sf=epochs.info["sfreq"],
        ch_names=epochs.info["ch_names"],
        band=bands,
        return_fit=True,
        win_sec=2 * (1 / bands[0]),
    )

    return freqs, psd_oscillatory
