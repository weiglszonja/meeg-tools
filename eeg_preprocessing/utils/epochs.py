import numpy as np
from matplotlib import pyplot as plt
from mne import Epochs, make_fixed_length_events, events_from_annotations
from mne.io import Raw
from mne.preprocessing import compute_current_source_density

from .config import settings
from .raw import filter_raw
from mne.utils import logger


def create_epochs(raw: Raw) -> Epochs:
    """
    Create non-overlapping segments from Raw instance.
    If there are annotations (trigger) found in the raw data, it creates
    epochs around those triggers i.e. an epoch is constructed from the
    start time of the current event to the end time of the next event.
    Parameters
    ----------
    raw: the continuous data to be segmented into non-overlapping epochs

    Returns
    -------
    Epochs instance
    """
    epoch_duration_in_seconds = settings['epochs']['duration']
    # remove slow drifts and high freq noise
    raw_bandpass = filter_raw(raw, n_jobs=8)
    if not bool(raw.annotations):
        logger.info('Creating epochs from continuous data ...')
        events = make_fixed_length_events(raw_bandpass,
                                          id=1,
                                          first_samp=True,
                                          duration=epoch_duration_in_seconds)
    else:
        logger.info('Creating epochs from annotations ...')
        events = get_events_from_annotations(raw_bandpass)

    epochs = Epochs(raw=raw_bandpass,
                    events=events,
                    picks='all',
                    event_id=list(np.unique(events[..., 2])),
                    baseline=None,
                    tmin=0.,
                    tmax=epoch_duration_in_seconds - (1 / raw.info['sfreq']),
                    preload=True)

    return epochs


def get_events_from_annotations(raw: Raw) -> np.ndarray:
    events, _ = events_from_annotations(raw)

    # preallocate array to store all events
    events_array = np.array([np.empty(3)], dtype=int)
    sfreq = raw.info['sfreq']
    duration = settings['epochs']['duration']

    for event_ind in range(len(events) - 1):
        current_event = events[event_ind]
        next_event = events[event_ind + 1]

        # events with shorter duration (less than the duration of the epochs)
        if next_event[0] / sfreq - current_event[0] / sfreq < duration:
            events_array = np.append(events_array, [events[event_ind]], axis=0)
            continue

        raw_segment = raw.copy().crop(tmin=current_event[0] / sfreq,
                                      tmax=next_event[0] / sfreq,
                                      include_tmax=True)

        epoch_events = make_fixed_length_events(raw_segment,
                                                id=int(current_event[2]),
                                                first_samp=True,
                                                duration=duration)

        events_array = np.append(events_array, epoch_events, axis=0)

    events_array = np.delete(events_array, [0], axis=0)

    return events_array


def compute_surface_laplacian(epochs: Epochs, show: True = bool,
                              **kwargs) -> Epochs:
    """
    Performs a surface Laplacian transform on the Epochs instance
    Additional arguments can be supplied, see the documentation for further
    details:
    https://mne.tools/dev/generated/mne.preprocessing.compute_current_source_density.html
    Parameters
    ----------
    epochs: the epochs to be transformed
    show: whether to visualize the power spectral densities before and after
    the Laplacian transform
    kwargs: optional keyword arguments

    Returns
    -------
    Raw instance
    """

    epochs_csd = compute_current_source_density(epochs, **kwargs)

    if show:
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex='all', sharey='all',
                               dpi=200)
        epochs.plot_psd(ax=ax[0], show=False)
        epochs_csd.plot_psd(ax=ax[1], show=False)
        plt.show()

    return epochs_csd
