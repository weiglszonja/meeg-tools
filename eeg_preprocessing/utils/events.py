from pathlib import Path
import mne
import pandas as pd


def get_events_from_raw(raw: mne.io.Raw) -> pd.DataFrame:
    raw_file_name = Path(raw.filenames[0]).stem
    subject, condition, day = raw_file_name.split('_')
    num_day = [d for d in day if d.isdigit()][0]

    # Read annotations from raw
    events_from_annot, _ = mne.events_from_annotations(raw)

    event_dict = {83: f'rs_{num_day}_1',
                  91: f'asrt_{num_day}_1',
                  93: f'asrt_{num_day}_2',
                  95: f'asrt_{num_day}_3',
                  97: f'asrt_{num_day}_4',
                  99: f'asrt_{num_day}_5',
                  101: f'asrt_{num_day}_6',
                  87: f'rs_{num_day}_2'}

    events_df = pd.DataFrame(data=events_from_annot, columns=['start_time', 'ignore', 'event_id']).drop(
        columns=['ignore'])
    events_df['start_time'] = events_df['start_time'] / raw.info['sfreq']
    events_df['event'] = events_df['event_id'].map(event_dict).fillna(events_df['event_id'])

    # blocks
    blocks = events_df.index[events_df['event_id'].isin([91, 93, 95, 97, 99, 101])]
    block_start_times = []
    for block_num in range(len(blocks)):
        block_start_index = blocks[block_num]
        block_end_index = events_df.index[events_df['event_id'].isin([52, 152])][block_num]
        block_df = events_df.iloc[block_start_index:block_end_index]
        assert len(block_df[block_df['event_id'].isin([19, 119])]) == 7
        real_block_start_time = block_df[block_df['event_id'].isin([19, 119])]['start_time'].values.tolist()[-1]
        block_start_times.append(real_block_start_time)

    block_events = events_df[events_df['event_id'].isin([91, 93, 95, 97, 99, 101])]
    #block_events['event'] = block_events['event_id'].map(event_dict).fillna(block_events['event_id'])
    block_events['real_start_time'] = block_start_times
    if num_day != '3':
        try:
            # only seq A
            block_events['end_time'] = events_df[events_df['event_id'] == 52]['start_time'].values
            block_events['sequence'] = 'A'
        except Exception as e:
            print(e)
            # only seq B
            block_events['end_time'] = events_df[events_df['event_id'] == 152]['start_time'].values
            block_events['sequence'] = 'B'
    else:
        # Az utolsó napon az 1-3-5 epochokban van az A és a 2-4-6 epochokban a B szekvencia a páratlan számú részvevőknél,
        # és a párosoknál forditva.
        if (int(subject) % 2) == 0:
            block_events.loc[block_events['event_id'].isin([91, 95, 99]), 'sequence'] = 'B'
            block_events.loc[block_events['event_id'].isin([93, 97, 101]), 'sequence'] = 'A'
        else:
            block_events.loc[block_events['event_id'].isin([91, 95, 99]), 'sequence'] = 'A'
            block_events.loc[block_events['event_id'].isin([93, 97, 101]), 'sequence'] = 'B'

        block_events['end_time'] = events_df[events_df['event_id'].isin([52, 152])]['start_time'].values

    # Create slice from resting events
    resting_events = events_df[events_df['event_id'].isin([83, 87])]
    resting_end_times = events_df[events_df['event_id'].isin([84, 88])]['start_time'].values

    resting_events['end_time'] = resting_end_times
    #resting_events['event'] = resting_events['event_id'].map(event_dict).fillna(resting_events['event_id'])

    return resting_events, block_events

