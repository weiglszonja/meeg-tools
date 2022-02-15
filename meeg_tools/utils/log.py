import os
from datetime import datetime
import re

import numpy as np
import pandas as pd

from mne import Epochs
from mne.utils import logger

from meeg_tools.utils.config import settings


def update_log(log_file_path: str, epochs: Epochs, notes: str) -> pd.DataFrame:
    """
    Documents the changes during preprocessing for an Epochs object.
    Custom description can be added with the notes argument.

    Parameters
    ----------
    log_file_path
    epochs
    notes

    Returns
    ----------
    log data
    """
    fid = epochs.info["temp"]
    dropped_epochs_marker = ["FASTER", "USER", "AUTOREJECT"]
    n_bad_epochs = len(
        [drop for drop in epochs.drop_log if np.isin(dropped_epochs_marker, drop).any()]
    )
    stimuli = list(epochs.event_id.keys())
    n_epochs_per_stimuli = [
        (", ").join([f"{ind}: {len(epochs[ind])}" for ind in stimuli])]

    log = pd.DataFrame(
        {
            "fid": [fid],
            "highpass": [epochs.info["highpass"]],
            "lowpass": [epochs.info["lowpass"]],
            "n_components": [np.NaN],
            "n_bad_epochs": [n_bad_epochs],
            "total_drop_percentage": [round(epochs.drop_log_stats(), 2)],
            "n_epochs_per_stimuli": n_epochs_per_stimuli,
            "stimuli": [list(epochs.event_id.keys())],
            "t_min": [epochs.tmin],
            "t_max": [epochs.tmax],
            "n_interpolated": [np.NaN],
            "average_ref_applied": [bool(epochs.info["custom_ref_applied"])],
            "baseline": [epochs.baseline if epochs.baseline else np.NaN],
            "notes": [notes],
            "date_of_update": [datetime.utcnow().isoformat()],
            "author": [settings["log"]["author"]]
        }
    )

    description = epochs.info["description"]

    if description is None:
        logger.warning(
            'Failed to update parameters from epochs.info["description"], \n'
            "Returning log object, consider adding description field manually and rerunning this function.\n"
            'Formatting should match this format: "n_components: 1, interpolated: AF7, Fp2, P3, CP5, Fp1"'
        )
        return log

    if "n_components" in description:
        n_components = [x for x in description.split(",")[0] if x.isdigit()][0]
        log["n_components"].update(int(n_components))

    if "interpolated" in description:
        description_plain = description.replace(" ", "").replace(":", "")
        interpolated_channels_str = description_plain.split("interpolated")[1]
        n_interpolated = len(interpolated_channels_str.split(","))
        log["n_interpolated"].update(n_interpolated)

    author_clean = re.sub("\W+", "", settings["log"]["author"])
    log_file_name = f"{author_clean}_log.csv"
    if os.path.isfile(os.path.join(log_file_path, log_file_name)):
        log.to_csv(os.path.join(log_file_path, log_file_name),
                   mode="a",
                   index=False,
                   header=False)
    else:
        log.to_csv(os.path.join(log_file_path, log_file_name),
                   index=False)

    return log
