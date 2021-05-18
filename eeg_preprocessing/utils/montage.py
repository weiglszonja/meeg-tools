from pathlib import Path
from mne.io import Raw
from mne.channels import read_custom_montage
from scipy import io
import pandas as pd


def set_raw_montage_from_locs(raw: Raw, montage_file_path: str):
    '''
    Reads channels locations from a file and applies them to Raw instance.
    Parameters
    ----------
    raw: the Raw instance with missing channel locations
    locs_file_path: the full path to the channel locations file (e.g. Starstim20.locs)

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
        raw.plot_sensors(show_names=True)

        return raw


def create_montage_from_mat_layout(mat_file_path: str):
    layout = io.loadmat(mat_file_path, squeeze_me=True)

    # create montage from channel positions
    montage = pd.DataFrame(data=layout['lay']['pos'].item(), columns=['Theta', 'Phi'])

    # add channel labels to montage
    num_channels = layout['lay']['label'].item().shape[0]
    channel_labels = [layout['lay']['label'].item()[channel] for channel in range(num_channels)]
    channel_map = dict(Gnd='AFz',
                       Ref='FCz')
    channel_labels = [channel_map.get(item, item) for item in channel_labels]
    montage['Site'] = channel_labels

    # reorder columns
    montage = montage[['Site', 'Theta', 'Phi']]

    # write montage to txt file
    layout_filename = Path(mat_file_path).stem
    layout_file_path = f'{layout_filename}.txt'
    if not Path(layout_file_path).is_file():
        montage.to_csv(layout_file_path, sep=' ', index=False, header=True)
    else:
        print(f'Montage already exists: {layout_file_path}')
