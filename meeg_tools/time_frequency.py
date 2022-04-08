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

from mne import Epochs, EvokedArray, pick_channels, read_evokeds, Evoked
from mne.channels import combine_channels, find_ch_adjacency
from mne.stats import combine_adjacency, permutation_cluster_1samp_test
from mne.time_frequency import tfr_morlet, AverageTFR, read_tfrs
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
                          picks=None,
                          combine='mean') -> pd.DataFrame:
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
    combine: whether to combine channels (defined by picks) using 'mean', 'median' or 'std'
    use None to not combine channels but get peak measures separately for each channel
    picks

    Returns
    -------
    DataFrame containing the peak measures for each channel or for a given ROI
    """
    if picks is None:
        picks = []

    erp_measures = pd.DataFrame(
        columns=[
            "fid",
            "ch_name",
            "tmin",
            "tmax",
            "mode",
            "peak_latency",
            "peak_amplitude",
        ]
    )

    if combine:
        picks_idx = pick_channels(erp.info["ch_names"], include=picks)
        roi_erp = combine_channels(erp, dict(roi=picks_idx), method=combine)
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
        for ch_name in picks:
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


def get_mean_amplitude(erp, tmin, tmax, mode):
    data = erp.crop(tmin=tmin, tmax=tmax).data
    sign_mean_data = data.squeeze()
    if mode == "pos":
        if not np.any(data > 0):
            logger.warning(
                f"{erp.comment} :" "No positive values encountered. Using default mode."
            )
        else:
            sign_mean_data = data[data > 0]
    elif mode == "neg":
        if not np.any(data < 0):
            logger.warning(
                f"{erp.comment} :" "No negative values encountered. Using default mode."
            )
        else:
            sign_mean_data = data[data < 0]
    elif mode == "abs":
        sign_mean_data = abs(sign_mean_data)

    mean_amplitude = sign_mean_data.mean(axis=0) * 1e6

    return mean_amplitude


def get_erp_measures_from_cross_condition_data(
    erp_arrays: List[EvokedArray],
    cross_condition_data: pd.DataFrame,
    interval_in_seconds: float,
):
    erp_measures = pd.DataFrame(
        columns=[
            "fid",
            "ch_name",
            "tmin",
            "tmax",
            "mode",
            "peak_latency",
            "peak_amplitude",
            "mean_amplitude",
        ]
    )

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

            mean_amp = get_mean_amplitude(erp=roi_erp, tmin=tmin, tmax=tmax, mode=mode)

            fid = Path(erp.comment.replace("\\", "/")).name
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
            for ch_name in cross_condition_data['ch_name']:
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

                mean_amp = get_mean_amplitude(erp=erp.copy().pick(ch_name),
                                              tmin=tmin,
                                              tmax=tmax,
                                              mode=mode)

                fid = Path(erp.comment.replace("\\", "/")).name
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


def read_tfrs_from_path(tfrs_path: str, pattern: str) -> List[AverageTFR]:
    """
    Reads TFR files from path with a given pattern to look for within the file name. I.e.
    used for separating conditions (e.g. H, L).
    Parameters
    ----------
    tfrs_path
    pattern

    Returns
    -------

    """
    files = sorted(
        [
            f
            for f in os.listdir(tfrs_path)
            if pattern.lower() in f.lower() and not f.startswith(".")
        ]
    )
    tfrs = [read_tfrs(os.path.join(tfrs_path, f))[0] for f in files]

    return tfrs


def read_evokeds_from_path(evokeds_path: str, pattern: str) -> List[Evoked]:
    """
    Reads Evoked files from path with a given pattern to look for within the file name. I.e.
    used for separating conditions (e.g. H, L).
    Parameters
    ----------
    evokeds_path
    pattern

    Returns
    -------

    """
    files = sorted(
        [
            f
            for f in os.listdir(evokeds_path)
            if pattern.lower() in f.lower() and not f.startswith(".")
        ]
    )
    evoked = [read_evokeds(os.path.join(evokeds_path, f), verbose=0)[0] for f in files]

    return evoked


def compute_power_difference(
    power_condition1: List[AverageTFR],
    power_condition2: List[AverageTFR],
    picks: [],
    baseline: None,
    tmin: float,
    tmax: float,
):

    if not picks:
        picks = None

    diff_over_participants = []
    for power1, power2 in zip(power_condition1, power_condition2):
        power1.pick_channels(picks).apply_baseline(baseline).crop(tmin=tmin, tmax=tmax)
        power2.pick_channels(picks).apply_baseline(baseline).crop(tmin=tmin, tmax=tmax)
        difference_data = (power1.data * 1e6) - (power2.data * 1e6)

        diff_over_participants.append(difference_data[np.newaxis, ...])

    diff_data = np.concatenate(diff_over_participants, axis=0)

    return diff_data


def permutation_correlation(diff_data, info, n_permutations, p_value):
    sensor_adjacency, ch_names = find_ch_adjacency(info, "eeg")
    adjacency = combine_adjacency(
        sensor_adjacency, diff_data.shape[2], diff_data.shape[3]
    )

    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
        diff_data,
        n_permutations=n_permutations,
        threshold=None,
        tail=0,
        adjacency=adjacency,
        out_type="mask",
        verbose=True,
    )

    # Create new stats image with only significant clusters for plotting
    T_obs_plot = np.nan * np.ones_like(T_obs)
    for c, p_val in zip(clusters, cluster_p_values):
        if p_val <= p_value:
            print(f"Significant cluster with p-value {p_val}")
            T_obs_plot[c] = T_obs[c]

    return T_obs_plot, cluster_p_values[cluster_p_values <= p_value]
