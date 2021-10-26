import os
import re

import numpy as np
from mne import Epochs
from mne.time_frequency import tfr_morlet, AverageTFR
from mne.utils import logger

from .utils.config import analysis


def compute_power(epochs: Epochs):
    """
    Computes Time-Frequency Representation (TFR) using complex Morlet wavelets
    averaged over epochs. Power is written to an HDF5 file.
    Parameters
    ----------
    epochs
    """
    fmin = analysis['morlet']['fmin']
    fmax = analysis['morlet']['fmax']
    step = analysis['morlet']['step']
    freqs = np.logspace(*np.log10([fmin, fmax]), num=step)
    n_cycles = freqs / 2.

    power = tfr_morlet(epochs,
                       picks=analysis['picks'] if 'picks' in analysis else 'all',
                       freqs=freqs,
                       n_cycles=n_cycles,
                       use_fft=False,
                       return_itc=False,
                       decim=analysis['morlet']['decim'],
                       average=True,
                       n_jobs=-1)

    save_to_hdf5(power=power)


def save_to_hdf5(power: AverageTFR):
    if not os.path.exists(power.info["fid"].parent):
        os.makedirs(power.info["fid"].parent)

    power.info["fid"] = str(power.info["fid"])
    # replace floats in file name with integers
    floats = re.findall("[-+]?\d*\.\d+", power.info["fid"])
    if floats:
        for num in floats:
            power.info["fid"] = power.info["fid"].replace(num, str(int(float(num))))
    file_path_with_extension = f'{power.info["fid"]}_{analysis["method"]}-tfr.h5'

    logger.info(f'Saving power at {file_path_with_extension} ...')
    power.save(fname=str(file_path_with_extension), overwrite=True)
    logger.info('[FINISHED]')


def average_power_into_frequency_bands(power: AverageTFR) -> AverageTFR:
    """
    Computes average power in specific frequency ranges defined in config.py.
    Parameters
    ----------
    power

    Returns
    -------

    """
    bands = analysis['bands'].values()

    band_power_data_arr = []
    for band in bands:
        band_filter = np.logical_and(power.freqs >= float(band[0]), power.freqs < float(band[1]))
        if int(max(power.freqs)) == int(max(bands)[1]):
            band_filter[-1] = True
        band_data = power.data[:, band_filter, :].mean(axis=1)
        band_power_data_arr.append(band_data[np.newaxis, :])

    band_power_data = np.concatenate(band_power_data_arr, axis=0)
    band_power_data = np.transpose(band_power_data, (1, 0, 2))

    band_freqs = np.asarray([band[0] for band in bands])

    band_power = power.copy()
    band_power.data = band_power_data
    band_power.freqs = band_freqs

    return band_power
