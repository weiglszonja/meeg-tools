from pathlib import Path

from mne.channels import read_custom_montage
from mne.io import read_raw, read_raw_edf, Raw

from .config import settings
from mne.utils import logger


def read_raw_measurement(raw_file_path: str, **kwargs):
    """
    Read raw EEG file from the given path.
    Parameters
    ----------
    raw_file_path: the full path to the EEG file
    locs_file_path: the full path to the channel locations file (optional)

    Returns
    -------
    Raw instance
    """
    raw_file_path = Path(raw_file_path)
    extension = raw_file_path.suffixes

    try:
        raw = read_raw(str(raw_file_path), preload=False, verbose=True)
    except Exception as e:
        if '.edf' in str(e):
            raw = read_raw_edf(str(raw_file_path), preload=False, verbose=True)
        else:
            logger.error(f'Unsupported file type ({extension})')

            return

    # Session parameters
    raw_id = raw_file_path.stem
    raw.info.update(fid=raw_id)

    if not bool(raw.get_montage()):
        logger.info(f'Channel locations are missing from the file')
        try:
            path_to_locs = kwargs['locs_file_path']
            raw = set_raw_montage_from_locs(raw=raw,
                                            path_to_locs=path_to_locs,
                                            show_montage=False)
        except Exception as e:
            logger.error(e)
            return raw

    return raw


def set_raw_montage_from_locs(raw: Raw, path_to_locs: str,
                              show_montage: False = bool) -> Raw:
    '''
    Reads channels locations from a file and applies them to Raw instance.
    Parameters
    ----------
    raw: the Raw instance with missing channel locations
    path_to_locs: the full path to the channel locations file
    (e.g. Starstim20.locs)
    show_montage: whether to show channel locations in a plot

    Returns
    -------
    Raw instance
    '''
    if Path(path_to_locs).exists():
        montage = read_custom_montage(path_to_locs)

        # check if there are any channels not in raw instance
        missing_channel_locations = [ch_name for ch_name in raw.ch_names if
                                     ch_name not in montage.ch_names]
        if missing_channel_locations:
            logger.info(f'There are {len(missing_channel_locations)} channel '
                        f'positions not present in the '
                        f'{Path(path_to_locs).stem} file.')
            logger.info(f'Assuming these ({missing_channel_locations}) '
                        f'are not EEG channels, dropping them from Raw.')

            raw.drop_channels(missing_channel_locations)

        logger.info('Applying channel locations to Raw instance.')
        raw.set_montage(montage)

        if show_montage:
            raw.plot_sensors(show_names=True)

    else:
        logger.warning(f'Montage file {path_to_locs} path does not exist! '
                       f'Returning unmodified raw.')

    return raw


def filter_raw(raw: Raw, **kwargs) -> Raw:
    """
    Applies band-pass filtering to Raw instance.
    Parameters
    ----------
    raw: the raw data to be filtered
    kwargs: optional keyword arguments to be passed to mne.filter()

    Returns
    -------
    Raw instance
    """
    raw_bandpass = raw.load_data().copy().filter(
        l_freq=settings['bandpass_filter']['low_freq'],
        h_freq=settings['bandpass_filter']['high_freq'],
        n_jobs=8,
        **kwargs)

    return raw_bandpass
