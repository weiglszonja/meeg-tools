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
                    preload=False)

    return epochs



# def create_epochs_from_stimulus_intervals(raw: Raw,
#                                           intervals: List[tuple]) -> Epochs:
#     """
#     Create non-overlapping epochs from stimulus intervals i.e. pairs of
#     triggers in the data that define a longer period (e.g. resting).
#     Intervals should be supplied as lists of tuples, where the first element
#     of each tuple denotes the start time of an event and the second element
#     the end time of the event (e.g. intervals=[(83, 84), (87, 88)]).
#     Parameters
#     ----------
#     raw
#     intervals
#
#     Returns
#     -------
#     Epochs instance
#     """
#     events, _ = events_from_annotations(raw)
#
#     if int(raw.first_samp) != 0:
#         events[..., 0] = events[..., 0] - raw.first_samp
#
#     epoch_duration = settings['epochs']['duration']
#     intervals_epoch_events = []
#
#     for interval in intervals:
#         interval_events = events[np.isin(events[..., 2], interval)]
#         t_min = interval_events[0][0] / raw.info['sfreq']
#         t_max = interval_events[1][0] / raw.info['sfreq']
#
#         raw_segment = raw.copy().crop(tmin=t_min,
#                                       tmax=t_max,
#                                       include_tmax=True)
#
#         epoch_id = interval_events[0][2]
#         epoch_events = make_fixed_length_events(raw_segment,
#                                                 id=int(epoch_id),
#                                                 first_samp=False,
#                                                 duration=epoch_duration)
#
#         epoch_events[..., 0] = epoch_events[..., 0] + interval_events[0][0]
#         intervals_epoch_events.append(epoch_events)
#
#     interval_events_array = np.concatenate(intervals_epoch_events, axis=0)
#
#     epochs = Epochs(raw=raw,
#                     events=interval_events_array,
#                     picks='all',
#                     event_id=list(np.unique(interval_events_array[..., 2])),
#                     baseline=None,
#                     tmin=0.,
#                     tmax=epoch_duration - (1 / raw.info['sfreq']),
#                     preload=True)
#
#     return epochs

