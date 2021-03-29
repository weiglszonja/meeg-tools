from pathlib import Path

from mne.io import read_raw_brainvision


def read_raw(raw_file_path: str, add_info: bool = True):
    """
    Read raw EEG file from the given path.
    Currently only Brain Vision files (.vhdr) are supported, for more formats contact the developer.
    Parameters
    ----------
    raw_file_path: the full path to the EEG file
    add_info: whether to add info (e.g. subject, condition, day) to the raw instance

    Returns
    -------
    Raw instance
    """
    raw_file_path = Path(raw_file_path)
    suffix = raw_file_path.suffix
    if not suffix == '.vhdr':
        print(f'The selected file format ({suffix}) is not the expected Brain Vision format (.vhdr)'
              ', please select another file.')
        return

    # Load raw file from path
    raw = read_raw_brainvision(raw_file_path, preload=False, verbose=True)

    # Session parameters
    if add_info:
        raw_id = raw_file_path.stem
        id_split = raw_id.split('_')
        subject = id_split[0]
        condition = id_split[1]
        num_day = [x for x in id_split[-1] if x.isdigit()][0]

        raw.info.update(fid=raw_id,
                        subject=subject,
                        condition=condition,
                        num_day=num_day)

    return raw
