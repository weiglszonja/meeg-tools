from pathlib import Path
from scipy import io
import pandas as pd


def create_montage_from_mat_layout(mat_file_path):
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
    montage.to_csv(f'{layout_filename}.txt', sep=' ', index=False, header=True)


if __name__ == '__main__':
    create_montage_from_mat_layout('acticap-64ch-standard2.mat')