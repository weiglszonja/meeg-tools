from pathlib import Path
from mne.io import Raw
from mne.channels import read_custom_montage
from scipy import io
import pandas as pd


def set_raw_montage_from_locs(raw: Raw, montage_file_path: str, show_montage: False = bool):
    '''
    Reads channels locations from a file and applies them to Raw instance.
    Parameters
    ----------
    raw: the Raw instance with missing channel locations
    montage_file_path: the full path to the channel locations file (e.g. Starstim20.locs)
    show_montage: whether to show channel locations in a plot

    Returns
    -------
    Raw instance
    '''
    if Path(montage_file_path).exists():
        montage = read_custom_montage(montage_file_path)

        # check if there are any channels not in raw instance
        missing_channel_locations = [ch_name for ch_name in raw.ch_names if
                                     ch_name not in montage.ch_names]
        if missing_channel_locations:
            print(f'There are {len(missing_channel_locations)} channel positions '
                  f'not present in the {Path(montage_file_path).stem} file.')
            print(f'Assuming these ({missing_channel_locations}) are not '
                  f'EEG channels, dropping them from Raw instance.')

            raw.drop_channels(missing_channel_locations)

        print('Applying channel locations to Raw instance.')
        raw.set_montage(montage)

        if show_montage:
            raw.plot_sensors(show_names=True)

        return raw
