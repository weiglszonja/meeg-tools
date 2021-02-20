from pathlib import Path
from scipy import io
import pandas as pd


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
