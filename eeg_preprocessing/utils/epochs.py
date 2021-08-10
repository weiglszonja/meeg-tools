import numpy as np
from mne import Epochs, find_events, events_from_annotations, \
    make_fixed_length_events
from mne.io import Raw
from typing import List

from mne.utils import logger

from .config import settings


def create_epochs_from_events(raw: Raw, event_ids: List) -> Epochs:
    """
    Create non-overlapping segments from Raw data.
    Note that temporal filtering should be done before creating the epochs.
    If there are annotations (triggers) found in the raw data, it creates
    epochs with respect to the stimulus onset defined by start_time
    and end_time in the configuration file (config.py) in seconds.
    Parameters
    ----------
    raw: the continuous data to be segmented into non-overlapping epochs
    event_ids: the list of event ids to create epochs from


    Returns
    -------
    Epochs instance
    """

    try:
        events = find_events(raw)
    except ValueError:
        events, _ = events_from_annotations(raw)

    selected_events = events[np.isin(events[..., 2], event_ids)]
    logger.info('Creating epochs from selected events ...')
    epochs = Epochs(raw=raw,
                    events=selected_events,
                    picks='all',
                    event_id=list(np.unique(selected_events[..., 2])),
                    baseline=None,
                    tmin=settings['epochs']['start_time'],
                    tmax=settings['epochs']['end_time'],
                    preload=False)

    return epochs


def create_epochs(raw: Raw) -> Epochs:
    """
    Create non-overlapping segments from Raw data with a fixed duration.
    Note that temporal filtering should be done before creating the epochs.
    The duration of epochs is defined in the configuration file (config.py).
    Parameters
    ----------
    raw: the continuous data to be segmented into non-overlapping epochs

    Returns
    -------
    Epochs instance
    """

    epoch_duration_in_seconds = settings['epochs']['duration']
    logger.info('Creating epochs from continuous data ...')
    events = make_fixed_length_events(raw,
                                      id=1,
                                      first_samp=True,
                                      duration=epoch_duration_in_seconds)

    epochs = Epochs(raw=raw,
                    events=events,
                    picks='all',
                    event_id=list(np.unique(events[..., 2])),
                    baseline=None,
                    tmin=0.,
                    tmax=epoch_duration_in_seconds, #- (1 / raw.info['sfreq']
                    preload=True)

    return epochs
