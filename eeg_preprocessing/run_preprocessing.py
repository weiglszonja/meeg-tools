import os
import mne
from autoreject import autoreject, Ransac

from .utils.faster import faster_bad_epochs
import numpy as np
import pandas as pd
from random import sample


def concat_raws_from_events(events: pd.DataFrame, raw: mne.io.Raw) -> mne.io.Raw:
    raws = []
    for idx, event in events.iterrows():
        if 'real_start_time' in event:
            raws.append(raw.copy().crop(tmin=event['real_start_time'] + 5, tmax=event['end_time'], include_tmax=True))
        else:
            raws.append(raw.copy().crop(tmin=event['start_time'], tmax=event['end_time'], include_tmax=True))
    return mne.concatenate_raws(raws)


def run_ica(raw: mne.io.Raw) -> mne.preprocessing.ica:
    sfreq = raw.info['sfreq']
    # remove slow drifts and high freq noise
    raw_bandpass_ica = raw.load_data().copy().filter(l_freq=0.5, h_freq=45)

    # create fixed length epoch for preprocessing
    epoch_duration_in_seconds = 1.0

    events = mne.make_fixed_length_events(raw_bandpass_ica,
                                          id=1,
                                          first_samp=True,
                                          duration=epoch_duration_in_seconds)

    epochs = mne.Epochs(raw=raw_bandpass_ica,
                        events=events,
                        picks='eeg',
                        event_id=1,
                        baseline=None,
                        tmin=0.,
                        tmax=epoch_duration_in_seconds - (1 / sfreq),
                        preload=True)

    # remove from memory
    del raw_bandpass_ica

    print('Preliminary epoch rejection: ')
    bad_epochs = faster_bad_epochs(epochs, picks=None, thres=3, use_metrics=None)
    epochs_faster = epochs.copy().drop(bad_epochs, reason='FASTER')

    del epochs

    # run ica
    n_components = 32
    method = 'infomax'

    ica = mne.preprocessing.ICA(n_components=n_components, random_state=42, method=method)
    ica.fit(epochs_faster, decim=2)

    epochs_faster.set_channel_types({'Fp1': 'eog', 'Fp2': 'eog'})
    eog_indices, eog_scores = ica.find_bads_eog(epochs_faster)
    ica.exclude = eog_indices
    return ica, epochs_faster


def run_autoreject(epochs: mne.Epochs, n_jobs: int = 11, subset: bool = False) -> mne.Epochs:
    # run autoreject on epochs
    ar = autoreject.AutoReject(random_state=42, n_jobs=n_jobs)

    if subset:
        n_epochs = len(epochs)
        print('Fitting autoreject on random subset of epochs: ')
        subset = sample(set(np.arange(0, n_epochs, 1)), int(n_epochs * 0.25))
        ar.fit(epochs[subset])

    else:
        print('Fitting autoreject on epochs: ')
        ar.fit(epochs)

    reject_log = ar.get_reject_log(epochs)
    # epochs[reject_log.bad_epochs].plot(n_epochs=10, title='ALL BAD EPOCHS AUTOREJECT')

    reject_df = pd.DataFrame(data=reject_log.labels, columns=epochs.info['ch_names'])
    # 0 : good data segment
    # 1 : bad data segment not interpolated
    # 2 : bad data segment interpolated
    reject_df = reject_df.replace(1, 0)
    reject_df = reject_df.replace(2, 1)
    reject_df = reject_df.replace(np.nan, 0)

    # mark channels where more than 50% of epochs were marked as bad channels
    bad_channels = reject_df.astype(bool).sum(axis=0) > len(epochs) / 2
    bads = bad_channels.index[bad_channels].tolist()

    epochs.drop(reject_log.bad_epochs, reason='AUTOREJECT')
    epochs.info['bad_channels_autoreject'] = bads

    return epochs


def run_ransac(epochs: mne.Epochs, n_jobs: int = 11) -> mne.Epochs:
    # find bad channels with ransac
    ransac = Ransac(verbose='progressbar', n_jobs=n_jobs)
    epochs_ransac = ransac.fit_transform(epochs)
    if ransac.bad_chs_:
        bads_str = ', '.join(ransac.bad_chs_)
        epochs_ransac.info.update(description=epochs_ransac.info['description'] + ' interpolated: '+bads_str)

    return epochs_ransac

#
# if __name__ == '__main__':
#     # Set base path to EEG data
#     base_path = 'G:/TMS_rewiring/'
#     eeg_path = os.path.join(base_path, 'Raw_data/24_L/Day1/EEG/')
#     # Create folder for preprocessed and interim files
#     folder_name = 'preprocessed'
#     interim_path = os.path.join(base_path, folder_name)
#
#     raw_file_name = [file for file in os.listdir(eeg_path) if file.endswith('.vhdr')][0]
#     subject, condition, day = raw_file_name.split('_')
#     num_day = [d for d in day if d.isdigit()][0]
#
#     # read raw file (not loaded into memory)
#     raw = mne.io.read_raw_brainvision(os.path.join(eeg_path, raw_file_name), preload=False, verbose=True)
#
#     resting_events, block_events = get_events_from_raw(raw)
#
#     # Create path to interim epoch files
#     interim_epochs_path = os.path.join(interim_path, condition, 'epochs_rs')
#     if not os.path.exists(interim_epochs_path):
#         os.makedirs(interim_epochs_path)
#
#     # Create path to interim raw files
#     interim_raw_path = os.path.join(interim_path, condition, 'raw_rs')
#     if not os.path.exists(interim_raw_path):
#         os.makedirs(interim_raw_path)
#
#     raw_rs = concat_raws_from_events(resting_events)
#     epochs_ica = run_ica(raw=raw_rs)
#
#










