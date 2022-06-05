import os
import random
import scipy.io as sio
import numpy as np
import pandas as pd
import logging
from random import shuffle
random.seed(3333)

logger = logging.getLogger('Data Import')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def get_bci_iii_data(path, mode):
    labels = ['aa', 'al', 'av', 'aw', 'ay']
    data = list()
    for label in labels:
        result = sio.loadmat(f'{path}/{label}.mat')
        markers = sio.loadmat(f'{path}/true_labels_{label}.mat')
        true_labels = markers['true_y'][0]

        for idx, ml in enumerate(true_labels):
            result['mrk']['y'][0][0][0][idx] = ml

        #result['mrk']['y'][0][0][0][-12] = markers['true_y'][0][-12]
        #print(result['mrk']['y'][0][0][0])

        tmp = list()
        for idx, time in enumerate(result['mrk']['pos'][0][0][0]):
            start_idx = int(time + 50)
            end_idx = int(start_idx + 200)
            tmp.append(result['cnt'][start_idx:end_idx, :])
        data.append([np.array(tmp), result['mrk']['y'][0][0][0]])

    final = dict(zip(labels, data))

    logger.info(f'*** Import of BCI Competition III IVa dataset done ***\n\n')
    return final[mode]


def get_npz_data(path, user, labels=None, sec=3):
    assert sec <= 3
    full_path = f'{path}/{user}.npz'
    if labels is None:
        labels = [769, 770, 771, 772]

    logger.info(f'Fetching npz data from {full_path}\n')
    bci_competition_data = np.load(full_path)

    if user[-1] == 'E':
        true_labels = np.ndarray.flatten(sio.loadmat(f'{path}/true_labels/{user}.mat')['classlabel'])
        new_labels = np.zeros(shape=true_labels.shape)
        mapping = dict(zip([1, 2, 3, 4], [769, 770, 771, 772]))
        for idx, label in enumerate(true_labels):
            new_labels[idx] = mapping[label]

        c = 0
        new_events = bci_competition_data['etyp'].copy()
        for idx, label in enumerate(bci_competition_data['etyp']):
            if label == 783:
                new_events[idx] = new_labels[c]
                c += 1
    else:
        new_events = bci_competition_data['etyp']

    raw_eeg = bci_competition_data['s'][:, :-3]
    raw_markers = zip(new_events, bci_competition_data['epos'], bci_competition_data['edur'])
    frame_length = sec*250

    tmp = list()
    for label, idx, _ in raw_markers:
        if label in labels:
            tmp.append(np.array([label[0], idx[0]+250, frame_length]))
        else:
            pass
    markers = np.array(tmp)

    tmp_b, tmp_m = list(), list()
    tmp_d = np.zeros(shape=(len(markers), frame_length, 22))

    for counter, (label, idx, duration) in enumerate(markers):
        tmp_d[counter] = raw_eeg[idx:idx+duration, :]
        tmp_b.append(label)
        epochs = np.empty(duration); epochs.fill(label)
        tmp_m.append(epochs)

    targets_per_timepoint = np.concatenate(tmp_m[:-1])

    data_per_batch = tmp_d
    targets_per_batch = np.array(tmp_b)
    logger.info(f'Import of BCI Competition IV 2a dataset done\nShape of data {data_per_batch.shape} and targets {targets_per_batch.shape}\n')

    return data_per_batch, targets_per_batch, targets_per_timepoint


def _check_length(marker_data):
    timestamps, deltas = list(), list()
    for idx, (_, (timestamp, _)) in enumerate(marker_data.iterrows()):
        timestamps.append(np.array(timestamp, dtype=np.float))
        if idx == 0:
            pass
        else:
            deltas.append(int(timestamps[idx] - timestamps[idx-1]))

    major_delta = max(set(deltas), key=deltas.count) - 1

    return major_delta, int(major_delta*256)

def read_single_data(path, labels, seconds, offset):
    eeg_keys = ['EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7', "EEG.P7", "EEG.O1",
                "EEG.O2", "EEG.P8", "EEG.T8", "EEG.FC6", "EEG.F4", "EEG.F8", "EEG.AF4"]
    marker_keys = ['Timestamp', 'MarkerValueInt']

    eeg_data = pd.read_csv(path, skiprows=[0]).loc[:, eeg_keys]
    marker_data_tmp = pd.read_csv(path, skiprows=[0]).loc[:, marker_keys].dropna().drop_duplicates(subset=['MarkerValueInt'])
    marker_data = marker_data_tmp[marker_data_tmp.MarkerValueInt.isin(labels)]
    num_samples = 6*256
    sample_length = seconds*256
    sample_offset = int(256*offset)

    msg = 'Sample length and offset longer then actual recording'
    assert num_samples >= (sample_offset + sample_length), logger.error(msg=msg)
    tmp_d, tmp_m, tmp_b = list(), list(), list()
    for idx, (marker_timestamp, (_, marker_value)) in enumerate(marker_data.iterrows()):
        'To increase the quality of shorter sample, we shift it towards the middle of the whole segment'
        if (sample_length == num_samples) or (offset >= 0.0):
            fmarker_timestamp = marker_timestamp
        else:
            fmarker_timestamp = marker_timestamp + (num_samples - sample_length)//2

        start_idx = fmarker_timestamp + sample_offset
        end_idx = start_idx + sample_length
        assert end_idx <= len(eeg_data)
        chunk = eeg_data.iloc[start_idx: end_idx]
        epochs = np.empty(sample_length); epochs.fill(marker_value)

        tmp_d.append(chunk)
        tmp_b.append(marker_value)
        tmp_m.append(epochs)

    raw_data_per_epoch = np.vstack(tmp_d)
    target_per_epoch = np.concatenate(tmp_m)

    raw_data_per_batch = np.array(tmp_d)
    target_per_batch = np.vstack(tmp_b)

    msg = f'Data import failed. Different shapes for data {raw_data_per_batch.shape} and target {target_per_batch.shape}.\n'
    assert len(raw_data_per_batch) == len(target_per_batch), logger.error(msg=msg)
    return raw_data_per_batch, target_per_batch, raw_data_per_epoch, target_per_epoch


def get_prev_project_data(path,  labels, seconds, offset=True):
    assert seconds <= 8
    if labels is None:
        labels = [0, 1, 2, 3]

    list_of_eeg_files = [f'{path}/{file}' for file in os.listdir(path) if '.csv' and 'EEG' in file]
    list_of_marker_files = [f'{path}/{file}' for file in os.listdir(path) if '.csv' and 'MARKERS' in file]

    path_pairs = list()
    for eeg_file in list_of_eeg_files:
        for marker_file in list_of_marker_files:
            if marker_file[-7:] == eeg_file[-7:]:
                path_pairs.append((eeg_file, marker_file))
            else:
                pass

    eeg_keys = ['EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7', "EEG.P7", "EEG.O1",
                "EEG.O2", "EEG.P8", "EEG.T8", "EEG.FC6", "EEG.F4", "EEG.F8", "EEG.AF4"]

    tmp_d, tmp_m = list(), list()
    frames = seconds*256

    for eeg_path, marker_path in path_pairs:
        logger.info(f'Fetching {eeg_path}\n')
        raw_markers = pd.read_csv(marker_path)
        markers = raw_markers[raw_markers.Value.isin(labels)]
        raw_eeg = pd.read_csv(eeg_path, skiprows=[0])

        for idx, (start_idx, end_idx, label, lvalue) in markers.iterrows():
            if end_idx-start_idx >= seconds:
                if offset and seconds < 7:
                    eeg_trial = raw_eeg.loc[raw_eeg['Timestamp'] > start_idx + 0.5].iloc[:frames].loc[:, eeg_keys]
                else:
                    eeg_trial = raw_eeg.loc[raw_eeg['Timestamp'] > start_idx].iloc[:frames].loc[:, eeg_keys]
                tmp_d.append(eeg_trial)
                tmp_m.append(lvalue)
            else:
                pass

    RAW_EEG = np.array(tmp_d)
    TARGET = np.array(tmp_m)

    logger.info(f'*** Import done ***\n'
                f'RAW EEG Data of shapes: {RAW_EEG.shape}\n'
                f'TARGET of shape: {TARGET.shape}\n')

    return RAW_EEG, TARGET


def get_project_data(path, labels=None, trials=-1, sec=5, offset=0.0):
    if labels is None:
        labels = [0, 1, 2, 3]

    list_of_files = [f'{path}/{file}' for file in os.listdir(path) if '.csv' in file][:trials]
    logger.info(f'{len(list_of_files)} files found.\n')
    shuffle(list_of_files)

    data_list_batch, target_list_batch = list(), list()
    data_list_epoch, target_list_epoch = list(), list()

    for idx, file in enumerate(list_of_files):
        eeg_data = pd.read_csv(file, skiprows=[0])
        duration = len(eeg_data)/256

        if duration > 25:
            logger.info(f'Fetching from {file}\nDuration {duration}s\n')
            try:
                data_batch, target_batch, data_epoch, target_epoch = read_single_data(file, labels, sec, offset)
                data_list_batch.append(data_batch)
                data_list_epoch.append(data_epoch)
                target_list_batch.append(target_batch)
                target_list_epoch.append(target_epoch)


            except AssertionError:
                logger.info(f'Going on with next sample...')

        else:
            logger.info(f'Discarding {path}\nDuration {duration}s < 25s\n')

    RAW_EEG = np.vstack(data_list_batch)
    TARGET = np.ndarray.flatten(np.vstack(target_list_batch))

    logger.info(f'*** Import done ***\n'
                f'RAW EEG Data of shapes: {RAW_EEG.shape}\n'
                f'TARGET of shape: {TARGET.shape}\n')


    return RAW_EEG, TARGET.astype(int)