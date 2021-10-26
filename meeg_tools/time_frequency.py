import os
import re

import numpy as np
from mne import Epochs
from mne.time_frequency import tfr_morlet, AverageTFR
from mne.utils import logger

from .utils.config import analysis


def compute_power(epochs: Epochs) -> AverageTFR:
    """
    Computes Time-Frequency Representation (TFR) using complex Morlet wavelets
    averaged over epochs.
    Returns an AverageTFR object with a shape of (n_channels, n_frequencies, n_times).
    Power is written to an HDF5 file.
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
