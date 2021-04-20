# Alternatively you can run this script instead of the data_preprocessing notebook

import scipy.signal
import numpy as np
from pymatreader import read_mat
import os

# Create a list with all the dataset files
dataset_files = []
dataset_folder = 'dataset'  # dataset folder to load data
for file in os.listdir(dataset_folder):
    if file == '__init__.py':
        continue  # skip init.py file
    dataset_files.append(os.path.join(dataset_folder, file))  # append relative path


def preprocess_data(mat):
    """
    This function is a Matlab rewritten function from
    https://gitlab.stimulate.ovgu.de/christoph.reichert/visual-spatial-attention-bci
    :param mat: matrix loaded with read_mat library
    :return: features extracted from the specific file
    """
    bci_exp = mat['bciexp']  # reference to the bci exp data which are the only relevant data
    labels = bci_exp['label']  # all channels

    sampling_rate = bci_exp['srate']
    downsampling_fact = 5  # downsampling factor, 250 / 5 = 50 Hz
    bandpass = np.array([1, 12.5])  # cutoff frequencies for the bandpass filter
    interval_len = .75  # seconds

    # calculate bandpass filter coefficients
    butter_b, butter_a = scipy.signal.butter(N=2, Wn=bandpass / (sampling_rate / 2), btype='bandpass')

    channels_of_interest = ['O9', 'CP1', 'CP2', 'O10', 'P7', 'P3', 'Pz',
                            'P4', 'P8', 'PO7', 'PO3', 'Oz', 'PO4', 'PO8']

    # get the index of each channel in labels array
    channel_indexes = np.array([labels.index(ch) for ch in channels_of_interest])
    lmast_channel_idx = np.char.equal('LMAST', labels)

    # number of samples per analysis window
    num_samples_per_window = int(interval_len * sampling_rate / downsampling_fact) - 1

    stimuli = np.array(bci_exp['stim'], dtype=np.double)
    num_stimuli = np.sum(np.diff(np.sum(stimuli[:, :, 0], axis=0)) > 0)

    eeg_data = bci_exp['data']
    num_trials = eeg_data.shape[2]
    num_channels = len(channel_indexes)
    data = np.zeros(shape=(num_channels, num_samples_per_window, num_stimuli, num_trials))

    # For each trial
    for tr in range(num_trials):
        rdat = eeg_data[channel_indexes, :, tr] - (eeg_data[lmast_channel_idx, :, tr] / 2)
        filtfilt_signal = scipy.signal.filtfilt(butter_b, butter_a, rdat, padtype='odd',
                                                padlen=3 * (max(len(butter_b), len(butter_a)) - 1))
        # Resample to 50 Hz
        rdat = scipy.signal.resample_poly(filtfilt_signal, 1, downsampling_fact, axis=1)
        rdat = rdat.T
        x1 = np.insert(stimuli[0, :, tr], 0, 0, axis=0)
        x2 = np.insert(stimuli[1, :, tr], 0, 0, axis=0)
        stim = np.array((np.diff(x1) > 0) + (np.diff(x2) > 0), dtype=np.double)

        stim_onsets = np.array(np.where(stim != 0))[0]

        for st in range(num_stimuli):
            start = int(np.ceil((stim_onsets[st] + 1) / downsampling_fact)) - 1
            idx = np.arange(start, start + num_samples_per_window)

            data[:, :, st, tr] = rdat[idx, :].T

    return data


def form_dataset(dataset_files):
    """
    Forms the dataset as a pair of features and labels
    :param dataset_files: array of all dataset files - their relative or absolute paths
    :return: pair of features and labels for ML
    """
    X_men, X_women, Y_men, Y_women, X, Y = [], [], [], [], [], []
    for file in dataset_files:
        mat = read_mat(file)
        data = preprocess_data(mat)
        features = data.reshape(-1, 14, 36, 10)
        labels = mat['bciexp']['intention']
        X.append(features)
        Y.append(labels)
        gender = mat['subject']['sex']

        if gender == 'male':
            X_men.append(features)
            Y_men.append(labels)
        else:
            X_women.append(features)
            Y_women.append(labels)

    X_concat = np.concatenate(X, axis=0)
    Y_concat = np.concatenate(Y)  # concatenate all labels
    X_men = np.concatenate(X_men)
    Y_men = np.concatenate(Y_men)
    X_women = np.concatenate(X_women)
    Y_women = np.concatenate(Y_women)

    print('The entire dataset X shape:', X_concat.shape)
    print('The entire dataset Y shape:', Y_concat.shape)
    print('Women dataset X shape:', X_women.shape)
    print('Women dataset Y shape:', Y_women.shape)
    print('Men dataset Y shape:', X_men.shape)
    print('Men dataset Y shape:', Y_men.shape)
    return X, Y, X_concat, Y_concat, X_men, Y_men, X_women, Y_women


# Concatenated dataset
X, Y, X_concat, Y_concat, X_men, Y_men, X_women, Y_women = form_dataset(dataset_files)

# Dataset path is dataset_result/bci_dataset.npz relative to this notebook
dataset_path = os.path.join('dataset_result')
concat_path = os.path.join(dataset_path, 'entire_dataset.npz')
men_only_path = os.path.join(dataset_path, 'dataset_male_gender.npz')
women_only_path = os.path.join(dataset_path, 'dataset_female_gender.npz')

# Create a directory if it does not exist already
os.makedirs('dataset_result', exist_ok=True)
np.savez_compressed(concat_path, features=X_concat, labels=Y_concat)
np.savez_compressed(men_only_path, features=X_men, labels=Y_men)
np.savez_compressed(women_only_path, features=X_women, labels=Y_women)

# Save each participant individually
for i in range(len(X)):
    file_path = os.path.join(dataset_path, 'P{:02d}'.format(i + 1))
    np.savez_compressed(file_path, features=X[i], labels=Y[i])

print('All saved')
