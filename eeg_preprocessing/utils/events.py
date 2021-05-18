from pathlib import Path
import mne
import pandas as pd
import numpy as np
from pandas.core.common import SettingWithCopyWarning
import warnings
from utils.config import config

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


def get_events_from_raw(raw: mne.io.Raw) -> pd.DataFrame:
    """
    Extract events from Raw data instance.

    The number of block start times and end times should be equal.
    In case of mismatch it returns a warning message and the extracted events unmodified.
    Parameters
    ----------
    raw: Raw instance to extract events from

    Returns
    -------
    The extracted block and resting events with their respective start and end times in seconds.
    """
    raw_file_name = Path(raw.filenames[0]).stem
    subject, condition, day = raw_file_name.split('_')
    num_day = [d for d in day if d.isdigit()][0]

    # Read annotations from raw
    events_from_annot, _ = mne.events_from_annotations(raw, verbose=False)

    event_dict = {83: f'rs_{num_day}_1',
                  87: f'rs_{num_day}_2',
                  91: f'asrt_{num_day}_1',
                  93: f'asrt_{num_day}_2',
                  95: f'asrt_{num_day}_3',
                  97: f'asrt_{num_day}_4',
                  99: f'asrt_{num_day}_5',
                  101: f'asrt_{num_day}_6',
                  19: 'practice_stimuli_seq_a',
                  119: 'practice_stimuli_seq_b',
                  52: 'last_isi_seq_a',
                  152: 'last_isi_seq_b'}

    events = pd.DataFrame(data=events_from_annot,
                          columns=['start_time', 'ignore', 'event_id']).drop(columns=['ignore'])
    events['start_time'] = events['start_time'] / raw.info['sfreq']
    events['event'] = events['event_id'].map(event_dict).fillna(events['event_id'])

    # Block events
    block_start_indices = events.index[events['event_id'].isin([91, 93, 95, 97, 99, 101])]
    block_end_indices = events.index[events['event_id'].isin([52, 152])]
    if len(block_start_indices) != len(block_end_indices):
        print(
            'There is a mismatch between the number of block start and end times!')
        return events

    block_start_times = []
    for start, end in zip(block_start_indices, block_end_indices):
        block = events.loc[start:end]
        if len(block[block['event_id'].isin([19, 119])]) != 7:
            print(
                'There is a mismatch between the number of practice blocks for at least one trial!')
            return events

        # take the last practice stimuli time as block start time
        offset_in_seconds = 5
        real_block_start_time = \
            block[block['event_id'].isin([19, 119])]['start_time'].values.tolist()[-1]
        block_start_times.append(real_block_start_time + offset_in_seconds)

    block_events = events.loc[block_start_indices]
    block_events['start_time'] = block_start_times

    if num_day != '3':
        block_events['sequence'] = events.loc[block_end_indices]['event_id'].map(
            {52: 'A', 152: 'B'}).values
    else:
        if (int(subject) % 2) == 0:
            block_events.loc[block_events['event_id'].isin([91, 95, 99]), 'sequence'] = 'B'
            block_events.loc[block_events['event_id'].isin([93, 97, 101]), 'sequence'] = 'A'
        else:
            block_events.loc[block_events['event_id'].isin([91, 95, 99]), 'sequence'] = 'A'
            block_events.loc[block_events['event_id'].isin([93, 97, 101]), 'sequence'] = 'B'

    block_events['end_time'] = events.loc[block_end_indices, 'start_time'].values

    # Resting events
    resting_events = events[events['event_id'].isin([83, 87])]
    resting_events['end_time'] = events.loc[events['event_id'].isin([84, 88]), 'start_time'].values

    # Append resting events to block events
    block_events = block_events.append(resting_events)

    # Add duration of events
    block_events['duration'] = block_events['end_time'] - block_events['start_time']

    return block_events


def create_epochs_from_events(raw: mne.io.Raw, events: pd.DataFrame):
    """
    Creates fixed length continuous epochs for ICA based on events.


    Parameters
    ----------
    raw: the instance to be used for creating the epochs
    events: the events to be used for creating the

    Returns
    -------
    Epochs instance
    """
    epoch_duration_in_seconds = config['epochs']['duration']

    # preallocate array to store all events
    events_array = np.array([np.empty(3)], dtype=int)

    for idx, event in events.sort_values('start_time').reset_index().iterrows():
        raw_segment = raw.copy().crop(tmin=event['start_time'],
                                      tmax=event['end_time'],
                                      include_tmax=True)

        epoch_events = mne.make_fixed_length_events(raw_segment,
                                                    id=event['event_id'],
                                                    first_samp=True,
                                                    duration=epoch_duration_in_seconds)

        events_array = np.append(events_array, epoch_events, axis=0)

    # remove pre-allocated row
    events_array = np.delete(events_array, [0], axis=0)

    # remove slow drifts and high freq noise
    raw_bandpass = raw.load_data().copy().filter(l_freq=config['bandpass_filter']['low_freq'],
                                                 h_freq=config['bandpass_filter']['high_freq'])

    epochs = mne.Epochs(raw=raw_bandpass,
                        events=events_array,
                        picks='eeg',
                        event_id=dict(zip(events['event'], events['event_id'])),
                        baseline=None,
                        tmin=0.,
                        tmax=epoch_duration_in_seconds - (1 / raw.info['sfreq']),
                        reject_by_annotation=True,
                        preload=True)

    # remove from memory
    del raw_bandpass

    return epochs
