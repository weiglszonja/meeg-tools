"""
This module contains functions that can be used to perform time-frequency analysis on
EEG/MEG data with MNE-Python.
"""
import os
import re
from typing import List
from pathlib import Path

import numpy as np
import pandas as pd

from mne import Epochs, EvokedArray, pick_channels
from mne.channels import combine_channels
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
        use_fft=True,
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


def compute_erp(epochs: Epochs, config: dict) -> EvokedArray:
    evoked = epochs.average(method=config["method"])

    return evoked


def get_erp_peak_measures(erp: EvokedArray,
                          tmin: float,
                          tmax: float,
                          mode: str,
                          picks=None, ) -> pd.DataFrame:
    """
    Computes peak measures (peak latency, peak amplitude) from Evoked instance for a
    given time interval defined by tmin and tmax in seconds. Peak measures can be
    computed for ROIs (averages data over given list of channels) by defining the list
    of channel names (e.g. ['F2', 'F5', 'F7']) that will be aggregated first,
    and then the measures computed.
    If picks is not defined, peak measures are computed for each channel.
    Parameters
    ----------
    erp
    tmin
    tmax
    mode: 'pos': finds the peak with a positive voltage (ignores negative voltages)
    'neg': finds the peak with a negative voltage (ignores positive voltages)
    'abs': finds the peak with the largest absolute voltage regardless of sign (positive or negative)
    picks

    Returns
    -------
    DataFrame containing the peak measures for each channel or for a given ROI
    """
    if picks is None:
        picks = []

    erp_measures = pd.DataFrame(
        columns=['fid', 'ch_name', 'tmin', 'tmax', 'mode', 'peak_latency',
                 'peak_amplitude'])

    if picks:
        picks_idx = pick_channels(erp.info['ch_names'], include=picks)
        roi_erp = combine_channels(erp, dict(roi=picks_idx), method='mean')
        _, lat, amp = roi_erp.get_peak(ch_type='eeg',
                                       tmin=tmin,
                                       tmax=tmax,
                                       mode=mode,
                                       return_amplitude=True)

        ch_name = ' '.join(picks)
        erp_measures = erp_measures.append(dict(fid=erp.comment,
                                                ch_name=ch_name,
                                                tmin=tmin,
                                                tmax=tmax,
                                                mode=mode,
                                                peak_latency=lat,
                                                peak_amplitude=amp * 1e6),
                                           ignore_index=True)

    else:
        for ch_name in erp.ch_names:
            _, lat, amp = erp.copy().pick(ch_name).get_peak(ch_type='eeg',
                                                            tmin=tmin,
                                                            tmax=tmax,
                                                            mode=mode,
                                                            return_amplitude=True)

            erp_measures = erp_measures.append(dict(fid=erp.comment,
                                                    ch_name=ch_name,
                                                    tmin=tmin,
                                                    tmax=tmax,
                                                    mode=mode,
                                                    peak_latency=lat,
                                                    peak_amplitude=amp * 1e6),
                                               ignore_index=True)

    return erp_measures


def get_erp_measures_from_cross_condition_data(erp_arrays: List[EvokedArray],
                                               cross_condition_data: pd.DataFrame,
                                               interval_in_seconds: float):
    erp_measures = pd.DataFrame(
        columns=['fid', 'ch_name', 'tmin', 'tmax',
                 'mode', 'peak_latency', 'peak_amplitude', 'mean_amplitude'])

    picks = cross_condition_data['ch_name'].values[0].split()
    mode = cross_condition_data['mode'].values[0]
    for erp in erp_arrays:
        picks_idx = pick_channels(erp.info['ch_names'], include=picks)

        if len(picks_idx) > 1:
            roi_erp = combine_channels(erp, dict(roi=picks_idx), method='mean')
            tmin = cross_condition_data['peak_latency'].values[0] - interval_in_seconds
            tmax = cross_condition_data['peak_latency'].values[0] + interval_in_seconds

            if tmax > erp.tmax:
                tmax = erp.tmax
            if tmin < erp.tmin:
                tmin = erp.tmin

            _, lat, amp = roi_erp.get_peak(ch_type='eeg',
                                           tmin=tmin,
                                           tmax=tmax,
                                           mode=mode,
                                           return_amplitude=True)

            erp_data_crop = roi_erp.crop(tmin=tmin, tmax=tmax).data

            if mode == 'pos':
                sign_mean_data = erp_data_crop[erp_data_crop > 0]
            elif mode == 'neg':
                sign_mean_data = erp_data_crop[erp_data_crop < 0]
            elif mode == 'abs':
                sign_mean_data = abs(erp_data_crop)
            else:
                sign_mean_data = erp_data_crop

            mean_amp = sign_mean_data.mean(axis=0) * 1e6

            fid = Path(erp.comment).name
            ch_name = cross_condition_data['ch_name'].values[0]
            erp_measures = erp_measures.append(dict(fid=fid,
                                                    ch_name=ch_name,
                                                    tmin=tmin,
                                                    tmax=tmax,
                                                    mode=mode,
                                                    peak_latency=lat,
                                                    peak_amplitude=amp * 1e6,
                                                    mean_amplitude=mean_amp),
                                               ignore_index=True)

        else:
            for ch_name in erp.ch_names:
                tmin = cross_condition_data[
                           cross_condition_data['ch_name'] == ch_name][
                           'peak_latency'] - interval_in_seconds
                tmax = cross_condition_data[
                           cross_condition_data['ch_name'] == ch_name][
                           'peak_latency'] + interval_in_seconds

                tmin = tmin.values[0]
                tmax = tmax.values[0]

                if tmax > erp.tmax:
                    tmax = erp.tmax
                if tmin < erp.tmin:
                    tmin = erp.tmin

                _, lat, amp = erp.copy().pick(ch_name).get_peak(ch_type='eeg',
                                                                tmin=tmin,
                                                                tmax=tmax,
                                                                mode=mode,
                                                                return_amplitude=True)

                erp_data_crop = erp.copy().pick(ch_name).crop(tmin=tmin, tmax=tmax).data

                if mode == 'pos':
                    sign_mean_data = erp_data_crop[erp_data_crop > 0]
                elif mode == 'neg':
                    sign_mean_data = erp_data_crop[erp_data_crop < 0]
                elif mode == 'abs':
                    sign_mean_data = abs(erp_data_crop)
                else:
                    sign_mean_data = erp_data_crop

                mean_amp = sign_mean_data.mean(axis=0) * 1e6

                fid = Path(erp.comment).name
                erp_measures = erp_measures.append(dict(fid=fid,
                                                        ch_name=ch_name,
                                                        tmin=tmin,
                                                        tmax=tmax,
                                                        mode=mode,
                                                        peak_latency=lat,
                                                        peak_amplitude=amp * 1e6,
                                                        mean_amplitude=mean_amp),
                                                   ignore_index=True)

    return erp_measures
