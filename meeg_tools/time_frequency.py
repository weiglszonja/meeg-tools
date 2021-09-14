import os
import numpy as np
from mne import Epochs
from mne.time_frequency import tfr_morlet, AverageTFR
from mne.utils import logger

from .utils.config import tfr


def compute_power(epochs: Epochs, to_hdf5: bool) -> np.ndarray:
    """
    Computes Time-Frequency Representation (TFR) using complex Morlet wavelets
    averaged over epochs and frequency bands. Returns a multi-dimensional array
    where power values are stored with a shape of
    (n_condition, n_frequency_bands, n_channels, n_times).
    When to_hdf5 is True, AverageTFR objects are written to separate HDF5 files
    (1) power before applying baseline and crop per condition
    (2) power per frequency band
    Parameters
    ----------
    epochs
    to_hdf5: whether to save power to hdf5 file (separately for each condition)

    Returns
    -------
    The computed TFR with a shape of (n_condition, n_frequency_bands,
    n_channels, n_times).
    """
    fmin = tfr['morlet']['fmin']
    fmax = tfr['morlet']['fmax']
    step = tfr['morlet']['step']
    freqs = np.logspace(*np.log10([fmin, fmax]), num=step)
    n_cycles = freqs / 2.

    powers = []
    for event_num, event_id in enumerate(epochs.event_id):
        power = tfr_morlet(epochs[event_id],
                           freqs=freqs,
                           n_cycles=n_cycles,
                           use_fft=False,
                           return_itc=False,
                           decim=tfr['morlet']['decim'],
                           average=True,
                           n_jobs=8)
        power.comment = event_id

        if to_hdf5:
            save_to_hdf5(power=power)

        if tfr['baseline']['range'] is not None:
            power.apply_baseline(baseline=tfr['baseline']['range'],
                                 mode=tfr['baseline']['mode'])

            power.crop(tmin=0.0, tmax=power.times[-1])

        power_bands = []
        for band, freq in tfr['bands'].items():
            freq_ind = np.where(
                np.logical_and(freqs.round(2) >= freq[0],
                               freqs.round(2) <= freq[1]))
            power_per_band = np.squeeze(power.data[:, freq_ind, :])

            # uncomment to plot power per frequency band
            # band_freqs = power.freqs[freq_ind]
            # power_band = AverageTFR(info=power.info,
            #                         data=power_per_band,
            #                         times=power.times,
            #                         freqs=band_freqs,
            #                         nave=power.nave,
            #                         method=power.method,
            #                         comment=f'{power.comment}_{band}')
            # power_band.plot(title=power_band.comment)

            # uncomment to save a hdf5 file per frequency band
            # if to_hdf5:
            #     save_to_hdf5(power=power_band)

            power_band_average = power_per_band.mean(axis=1)
            power_bands.append(power_band_average[np.newaxis, ...])

        power_bands_data = np.concatenate(power_bands, axis=0)

        powers.append(power_bands_data[np.newaxis, ...])

    powers_per_condition = np.concatenate(powers, axis=0)

    return powers_per_condition


def save_to_hdf5(power: AverageTFR):
    method = power.method.replace('-', '_')

    fname = f'{power.info["fid"]}_{power.comment}_{method}'

    if not os.path.exists(power.info['fid'].parent):
        os.makedirs(power.info['fid'].parent)

    if not os.path.exists(f'{fname}-tfr.h5'):
        power.info['fid'] = str(power.info['fid'])
        power.save(fname=f'{fname}-tfr.h5', overwrite=False)
    else:
        logger.warning(f'The file {fname} already exists.')
