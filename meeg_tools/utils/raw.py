import os
from pathlib import Path

from mne import concatenate_raws
from mne.channels import read_custom_montage
from mne.io import read_raw, read_raw_edf, Raw
from mne.utils import logger

from .config import settings


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
        if ".edf" in str(e):
            raw = read_raw_edf(str(raw_file_path), preload=False, verbose=True)
        else:
            logger.error(f"Unsupported file type ({extension})")

            return

    # Session parameters
    raw_id = raw_file_path.stem
    raw.info.update(temp=raw_id)

    if not bool(raw.get_montage()):
        logger.info(f"Channel locations are missing from the file")
        try:
            path_to_locs = kwargs["locs_file_path"]
            raw = set_raw_montage_from_locs(
                raw=raw, path_to_locs=path_to_locs, show_montage=False
            )
        except Exception as e:
            logger.error(e)
            return raw

    logger.info(raw.info)

    return raw


def set_raw_montage_from_locs(
    raw: Raw, path_to_locs: str, show_montage: False = bool
) -> Raw:
    """
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
    """
    if Path(path_to_locs).exists():
        montage = read_custom_montage(path_to_locs)

        # check if there are any channels not in raw instance
        missing_channel_locations = [
            ch_name for ch_name in raw.ch_names if ch_name not in montage.ch_names
        ]
        if missing_channel_locations:
            logger.info(
                f"There are {len(missing_channel_locations)} channel "
                f"positions not present in the "
                f"{Path(path_to_locs).stem} file."
            )
            logger.info(
                f"Assuming these ({missing_channel_locations}) "
                f"are not EEG channels, dropping them from Raw."
            )

            raw.drop_channels(missing_channel_locations)

        logger.info("Applying channel locations to Raw instance.")
        raw.set_montage(montage)

        if show_montage:
            raw.plot_sensors(show_names=True)

    else:
        logger.warning(
            f"Montage file {path_to_locs} path does not exist! "
            f"Returning unmodified raw."
        )

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
    raw_bandpass = (
        raw.load_data()
        .copy()
        .filter(
            l_freq=settings["bandpass_filter"]["low_freq"],
            h_freq=settings["bandpass_filter"]["high_freq"],
            n_jobs=8,
            **kwargs,
        )
    )

    return raw_bandpass


def concat_raws_with_suffix(path_to_raw_files: str, suffix: str) -> Raw:
    """
    Concatenates raw measurement files with a given suffix (e.g. ".vhdr") from a folder.
    File namings should follow an order e.g. the first part of the measurement is
    "eeg_1.vhdr" and the second part is "eeg_1_2.vhdr".
    Returns the concatenated instance as if it was a continuous measurement.
    Parameters
    ----------
    path_to_raw_files: str
        The path to the folder where the raw files are located.
    suffix: str
        The name of the file extension (e.g. ".vhdr", ".edf")
    Returns
    -------
    Raw instance
    """

    raw_file_path = Path(path_to_raw_files)
    file_names_in_order = sorted(
        [f.stem for f in raw_file_path.rglob('*.*') if f.suffix == suffix])
    files = [str(f) for file_name in file_names_in_order for f in
             raw_file_path.rglob(f'{file_name}{suffix}')]
    logger.info(f"Found {len(file_names_in_order)} files with {suffix} extension:\n"
                f"{', '.join(files)}")
    raws = [read_raw(file, preload=False, verbose=True) for file in files]
    raw = concatenate_raws(raws)
    # Session parameters
    raw_id = file_names_in_order[0]
    raw.info.update(temp=raw_id)

    logger.info(raw.info)

    return raw
