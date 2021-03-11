from pathlib import Path
import mne
import pandas as pd
from pandas.core.common import SettingWithCopyWarning
import warnings

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
        real_block_start_time = \
            block[block['event_id'].isin([19, 119])]['start_time'].values.tolist()[-1]
        block_start_times.append(real_block_start_time)

    block_events = events.loc[block_start_indices]
    block_events['real_start_time'] = block_start_times

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

    return block_events
