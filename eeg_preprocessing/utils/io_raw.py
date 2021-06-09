from pathlib import Path

from mne.io import read_raw_brainvision, read_raw_edf, Raw
from mne import Epochs, make_fixed_length_events

from .config import settings


def read_raw(raw_file_path: str, add_info: bool = True):
    """
    Read raw EEG file from the given path.
    Parameters
    ----------
    raw_file_path: the full path to the EEG file
    add_info: to add info (e.g. subject, condition, day) to the raw instance

    Returns
    -------
    Raw instance
    """
    raw_file_path = Path(raw_file_path)
    suffix = raw_file_path.suffix

    # Load raw file from path
    if suffix == '.vhdr':
        raw = read_raw_brainvision(str(raw_file_path),
                                   preload=False,
                                   verbose=True)
    elif suffix == '.edf':
        raw = read_raw_edf(str(raw_file_path),
                           preload=False,
                           verbose=True)
    else:
        print(f'The selected file format ({suffix}) is not the expected '
              f'Brain Vision format (.vhdr) or standard EDF (.edf) format, '
              f'please select another file.')
        return

    # Session parameters
    raw_id = raw_file_path.stem
    raw.info.update(fid=raw_id)
    if add_info:
        id_split = raw_id.split('_')
        subject = id_split[0]
        condition = id_split[1]
        num_day = [x for x in id_split[-1] if x.isdigit()][0]

        raw.info.update(subject=subject,
                        condition=condition,
                        num_day=num_day)

    return raw


def create_epochs_from_raw(raw: Raw) -> Epochs:
    # remove slow drifts and high freq noise
    l_freq = settings['bandpass_filter']['low_freq']
    h_freq = settings['bandpass_filter']['high_freq']
    raw_bandpass = raw.load_data().copy().filter(l_freq=l_freq,
                                                 h_freq=h_freq)

    epoch_duration_in_seconds = settings['epochs']['duration']

    events = make_fixed_length_events(raw_bandpass,
                                      id=1,
                                      first_samp=True,
                                      duration=epoch_duration_in_seconds)

    epochs = Epochs(raw=raw_bandpass,
                    events=events,
                    picks='eeg',
                    event_id=1,
                    baseline=None,
                    tmin=0.,
                    tmax=epoch_duration_in_seconds - (1 / raw.info['sfreq']),
                    preload=True)

    # remove from memory
    del raw_bandpass

    return epochs
