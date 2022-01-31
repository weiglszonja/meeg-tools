from typing import List
import numpy as np
import pandas as pd

from mne import (
    Epochs,
    find_events,
    events_from_annotations,
    make_fixed_length_events,
    merge_events,
    concatenate_epochs,
)
from mne.epochs import combine_event_ids
from mne.io import Raw
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
    logger.info("Creating epochs from selected events ...")
    epochs = Epochs(
        raw=raw,
        events=selected_events,
        picks="all",
        event_id=list(np.unique(selected_events[..., 2])),
        baseline=None,
        tmin=settings["epochs"]["start_time"],
        tmax=settings["epochs"]["end_time"],
        preload=False,
    )

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

    epoch_duration_in_seconds = settings["epochs"]["duration"]
    logger.info("Creating epochs from continuous data ...")
    events = make_fixed_length_events(
        raw, id=1, first_samp=True, duration=epoch_duration_in_seconds
    )

    epochs = Epochs(
        raw=raw,
        events=events,
        picks="all",
        event_id=list(np.unique(events[..., 2])),
        baseline=None,
        tmin=0.0,
        tmax=epoch_duration_in_seconds,  # - (1 / raw.info['sfreq']
        preload=False,
    )

    return epochs


def create_epochs_from_intervals(raw: Raw, intervals: List[tuple]) -> Epochs:
    events, _ = events_from_annotations(raw)

    epochs_list = []
    for interval in intervals:
        start_idx = np.where(events[..., 2] == interval[0])[0]
        end_idx = np.where(events[..., 2] == interval[1])[0]

        raw_cropped = raw.copy().crop(
            tmin=events[start_idx[0]][0] / raw.info["sfreq"],
            tmax=events[end_idx[0]][0] / raw.info["sfreq"],
        )

        epochs = create_epochs(raw_cropped)
        combine_event_ids(epochs, list(epochs.event_id.keys()), interval[0], copy=False)

        epochs_list.append(epochs)

    return concatenate_epochs(epochs_list)


def create_metadata(epochs: Epochs):
    """
    Custom function that adds metadata to Epochs instance.
    Parameters
    ----------
    epochs

    Returns
    -------

    """
    metadata = pd.DataFrame(
        data=epochs.events, columns=["time_in_samples", "stim", "id"]
    )

    # we can add boundaries of epochs
    epoch_boundaries = [211, 212, 213, 214, 215, 216]

    edges = np.where(np.isin(epochs.events[..., 2], epoch_boundaries))[0]

    boundaries = dict(zip(epoch_boundaries, edges))
    logger.info("Found these indices for these epoch boundary events: ")
    logger.info("\n".join("{}\t{}".format(k, v) for k, v in boundaries.items()))
    for epoch_ind, epoch in enumerate(np.split(epochs.events, edges, axis=0)):
        if epoch_ind != len(edges):
            metadata.loc[metadata["time_in_samples"].isin(epoch[..., 0]), "epoch"] = (
                epoch_ind + 1
            )

    metadata.loc[
        metadata["id"].isin([10, 110, 11, 111, 14, 114, 15, 115]), "triplet"
    ] = "H"
    metadata.loc[
        metadata["id"].isin([12, 112, 13, 113, 16, 116, 112, 113, 116]), "triplet"
    ] = "L"
    metadata.loc[
        metadata["id"].isin(
            [10, 110, 14, 114],
        ),
        "triplet_type",
    ] = "HR"
    metadata.loc[
        metadata["id"].isin(
            [12, 112, 16, 116],
        ),
        "triplet_type",
    ] = "LR"
    metadata.loc[
        metadata["id"].isin(
            [11, 13, 15, 111, 113, 115],
        ),
        "triplet_type",
    ] = "P"
    metadata.loc[
        metadata["id"].isin([44, 45, 46, 47, 144, 145, 146, 147]), "answer"
    ] = "incorrect"
    metadata.loc[
        ~metadata["id"].isin([44, 45, 46, 47, 144, 145, 146, 147]), "answer"
    ] = "correct"

    metadata.loc[metadata["id"].isin([10, 11, 12, 13, 14, 15, 16]), "sequence"] = "A"
    metadata.loc[
        metadata["id"].isin([100, 111, 112, 113, 114, 115, 116]), "sequence"
    ] = "B"

    # find stimuli that are followed by an incorrect answer
    stimuli = metadata[
        metadata["id"].isin(
            [10, 110, 11, 111, 12, 112, 13, 113, 14, 114, 15, 115, 16, 116]
        )
    ]
    stimuli_indices = np.asarray(stimuli["id"].index.tolist())
    incorrect_indices = (
        np.asarray(metadata.index[metadata["answer"] == "incorrect"].tolist()) - 1
    )
    incorrect_answers = stimuli_indices[np.isin(stimuli_indices, incorrect_indices)]
    metadata.loc[incorrect_answers, "answer"] = "incorrect"

    return metadata


def create_new_events_from_metadata(epochs: Epochs) -> Epochs:
    # container for new events
    new_events_all = [np.zeros(3)]

    new_event_ids = {}
    event_counter = 0
    for triplet in epochs.metadata["triplet"].unique().tolist():
        for epoch in epochs.metadata["epoch"].unique().tolist():
            event_counter += 1
            epochs_filt = epochs[f"epoch == {int(epoch)} & triplet == '{triplet}'"]
            old_ids = list(epochs_filt.event_id.values())
            new_events = merge_events(
                events=epochs_filt.events,
                ids=old_ids,
                new_id=event_counter,
                replace_events=True,
            )
            new_event_ids[f"e{int(epoch)}_{triplet}"] = int(event_counter)
            new_events_all = np.concatenate([new_events_all, new_events], axis=0)

    new_events_all = new_events_all[1:]

    epochs.event_id = new_event_ids
    epochs.events = new_events_all
    return epochs
