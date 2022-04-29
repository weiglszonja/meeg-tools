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

    if isinstance(epochs.metadata, pd.DataFrame):
        metadata = epochs.metadata
    else:
        metadata = pd.DataFrame(
            data=epochs.events, columns=["time_in_samples", "stim", "id"]
        )

        # we can add boundaries of epochs
        epoch_boundaries = [210, 211, 212, 213, 214, 215, 216]
        practice_end_trigger = epoch_boundaries[0]

        edges = np.where(np.isin(epochs.events[..., 2], epoch_boundaries))[0]

        boundaries = dict(zip(epoch_boundaries, edges))
        logger.info("Found these indices for these epoch boundary events: ")
        logger.info("\n".join("{}\t{}".format(k, v) for k, v in boundaries.items()))
        for epoch_ind, epoch in enumerate(np.split(epochs.events, edges, axis=0)):
            if practice_end_trigger in boundaries:
                epoch_number = epoch_ind
            else:
                epoch_number = epoch_ind + 1
            metadata.loc[
                metadata["time_in_samples"].isin(epoch[..., 0]), "epoch"
            ] = epoch_number

        metadata.loc[
            metadata["id"].isin(
                [44, 45, 46, 47, 49, 67, 68, 69, 70, 78, 144, 145, 146, 147, 149]
            ),
            "answer",
        ] = "incorrect"
        metadata.loc[
            metadata["id"].isin(
                [40, 41, 42, 43, 63, 64, 65, 66, 140, 141, 142, 143]
            ),
            "answer",
        ] = "correct"

        # find stimuli that are followed by an incorrect answer
        stimuli = metadata[
            metadata["id"].isin(
                [
                    10,
                    110,
                    11,
                    111,
                    12,
                    112,
                    13,
                    113,
                    14,
                    114,
                    15,
                    115,
                    16,
                    116,
                    17,
                    117,
                    18,
                    118,
                    19,
                    119,
                    61,
                    62,
                ]
            )
        ]
        stimuli_indices = np.asarray(stimuli["id"].index.tolist())
        incorrect_indices = (
            np.asarray(metadata.index[metadata["answer"] == "incorrect"].tolist()) - 1
        )
        incorrect_answers = stimuli_indices[np.isin(stimuli_indices, incorrect_indices)]
        metadata.loc[incorrect_answers, "answer"] = "incorrect"

        correct_indices = (
            np.asarray(metadata.index[metadata["answer"] == "correct"].tolist()) - 1
        )
        correct_answers = stimuli_indices[np.isin(stimuli_indices, correct_indices)]
        metadata.loc[correct_answers, "answer"] = "correct"

        # find arrow directions for stimuli
        arrow_stimuli = dict(
            left=[40, 44, 140, 144, 63, 67],
            up=[41, 45, 141, 145, 64, 68],
            down=[42, 46, 142, 146, 65, 69],
            right=[43, 47, 143, 147, 66, 70],
        )

        for arrow in arrow_stimuli:
            metadata.loc[metadata["id"].isin(arrow_stimuli[arrow]), "arrow"] = arrow
            stimuli_before_arrow = (
                np.asarray(metadata.index[metadata["arrow"] == arrow].tolist()) - 1
            )
            arrow_answers = stimuli_indices[
                np.isin(stimuli_indices, stimuli_before_arrow)
            ]

            metadata.loc[arrow_answers, "arrow"] = arrow

    metadata.loc[
        metadata["id"].isin([10, 11, 14, 15, 112, 113, 114, 115]), "triplet"
    ] = "high"
    metadata.loc[metadata["id"].isin([12, 13, 16, 110, 111, 116]), "triplet"] = "low"
    metadata.loc[metadata["id"].isin([19, 119]), "triplet"] = "X"
    metadata.loc[metadata["id"].isin([17, 117]), "triplet"] = "trill"
    metadata.loc[metadata["id"].isin([18, 118]), "triplet"] = "repetition"
    metadata.loc[metadata["id"].isin([61, 62]), "triplet"] = "practice"
    metadata.loc[
        metadata["id"].isin(
            [10, 14, 112, 114],
        ),
        "triplet_type",
    ] = "high-random"
    metadata.loc[
        metadata["id"].isin(
            [12, 16, 110, 116],
        ),
        "triplet_type",
    ] = "low-random"
    metadata.loc[
        metadata["id"].isin(
            [11, 13, 15, 111, 113, 115],
        ),
        "triplet_type",
    ] = "pattern"

    metadata.loc[
        metadata["id"].isin([10, 11, 12, 13, 14, 15, 16, 17, 18, 19]), "sequence"
    ] = "A"
    metadata.loc[
        metadata["id"].isin([110, 111, 112, 113, 114, 115, 116, 117, 118, 119]),
        "sequence",
    ] = "B"

    metadata.loc[metadata["id"].isin([11, 13, 15, 111, 113, 115]), "stimuli"] = "pattern"

    metadata.loc[
        metadata["id"].isin([10, 110, 14, 114, 12, 112, 16, 116]), "stimuli"
    ] = "random"

    metadata.loc[metadata["id"].isin([10, 11, 110, 111]), "rewiring"] = "high-low"

    metadata.loc[metadata["id"].isin([12, 13, 112, 113]), "rewiring"] = "low-high"

    metadata.loc[metadata["id"].isin([14, 15, 114, 115]), "rewiring"] = "high-high"

    metadata.loc[metadata["id"].isin([16, 116]), "rewiring"] = "low-low"

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
