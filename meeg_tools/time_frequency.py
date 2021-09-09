import numpy as np
from mne import Epochs
from mne.time_frequency import tfr_morlet

from .utils.config import tfr


def compute_power(epochs: Epochs, to_hdf5: bool) -> np.ndarray:
    """
    Computes Time-Frequency Representation (TFR) using complex Morlet wavelets
    averaged over epochs. Returns a multi-dimensional array where power values
    are stored with a shape of (n_condition, n_signals, n_freqs, n_times).
    Parameters
    ----------
    epochs
    to_hdf5: whether to save power to hdf5 file (separately for each condition)

    Returns
    -------
    The computed TFR with a shape of (n_condition, n_signals, n_freqs, n_times)
    """
    fmin = tfr['morlet']['fmin']
    fmax = tfr['morlet']['fmax']
    step = tfr['morlet']['step']
    freqs = np.logspace(*np.log10([fmin, fmax]), num=step)
    n_cycles = freqs / 2.

    n_condition = len(epochs.event_id)
    n_channels = len(epochs.ch_names)
    n_freqs = len(freqs)
    n_times = len(epochs.times)
    powers = np.zeros((n_condition, n_channels, n_freqs, n_times))

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

        power.apply_baseline(baseline=tfr['baseline']['range'],
                             mode=tfr['baseline']['mode'])

        if 'fid' in epochs.info:
            fname = f'{epochs.info["fid"]}_{power.comment}_{power.method}'
        else:
            fname = f'{power.comment}_{power.method}'

        if to_hdf5:
            power.save(fname=fname + '-tfr.h5', overwrite=False)

        powers[event_num] = power.data

    return powers
