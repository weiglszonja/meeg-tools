import os
from datetime import datetime
import numpy as np
import pandas as pd
from mne import Epochs


def update_log(log_file_path: str, epochs: Epochs, notes: str) -> pd.DataFrame:
    """
    Documents the changes during preprocessing for an Epochs object.
    Custom description can be added with the notes argument.
    The file path has to contain the file extension too (.csv).

    Parameters
    ----------
    log_file_path
    epochs
    notes

    Returns
    ----------
    log data
    """
    fid = epochs.info['fid']
    dropped_epochs_marker = ['FASTER', 'USER', 'AUTOREJECT']
    n_bad_epochs = len([drop for drop in epochs.drop_log if
                        np.isin(dropped_epochs_marker, drop).any()])

    log = pd.DataFrame({'fid': [fid],
                        'highpass': [epochs.info['highpass']],
                        'lowpass': [epochs.info['lowpass']],
                        'n_components': [np.NaN],
                        'n_bad_epochs': [n_bad_epochs],
                        'n_total_epochs': [len(epochs)],
                        'drop_percentage': [round(epochs.drop_log_stats(), 2)],
                        'stimuli': [list(epochs.event_id.keys())],
                        't_min': [epochs.tmin],
                        't_max': [epochs.tmax],
                        'n_interpolated': [np.NaN],
                        'average_ref_applied': [
                            bool(epochs.info['custom_ref_applied'])],
                        'baseline': [
                            epochs.baseline if epochs.baseline else np.NaN],
                        'notes': [notes],
                        'date_of_update': [datetime.utcnow().isoformat()]})

    description = epochs.info['description']
    if 'n_components' in description:
        n_components = [x for x in description.split(',')[0] if x.isdigit()][0]
        log['n_components'].update(int(n_components))

    if 'interpolated' in description:
        description_plain = description.replace(' ', '').replace(':', '')
        interpolated_channels_str = description_plain.split('interpolated')[1]
        n_interpolated = len(interpolated_channels_str.split(','))
        log['n_interpolated'].update(n_interpolated)

    if os.path.isfile(log_file_path):
        log.to_csv(log_file_path, mode='a', index=False, header=False)
    else:
        log.to_csv(log_file_path, index=False)

    return log
