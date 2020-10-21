import logging
import os
from typing import Union, List, Tuple

import mne

import numpy as np


class Preprocessing:

    def __init__(self,
                 root_path=os.path.join('..', 'datasets', 'p300'),
                 mne_log=False,
                 t_pre_stimulus=.2,
                 t_post_stimulus=1.0):
        self.root_path = root_path
        self.mne_log = mne_log
        self.t_pre_stimulus = t_pre_stimulus
        self.t_post_stimulus = t_post_stimulus
        self.logger = logging.getLogger('Preprocessing')

    @staticmethod
    def check_if_dataset_exists(dataset_file_path=os.path.join('..', 'datasets', 'p300', 'p300_dataset')):
        return os.path.exists(dataset_file_path)

    def load_dataset(self, dataset_file_path=os.path.join('..', 'datasets', 'p300', 'p300_dataset')):
        data = np.load(file=dataset_file_path)
        try:
            return (data['train_x'], data['train_y']), (data['test_x'], data['test_y'])
        except KeyError:
            self.logger.error('Error, specified file does not contain dataset or is corrupt')

        return None

    def create_new_dataset(self, test_percent=.25, save=True,
                           output_filepath=os.path.join('..', 'datasets', 'p300', 'p300_dataset')):
        if 0 > test_percent > .5:
            self.logger.info('Invalid dataset test percentage - below 0 or above 50%, setting to 25%')
            test_percent = .25

        if not self.mne_log:
            mne.set_log_level(verbose=False)

        raw_list = self.__get_vhdr_file_list__()
        epochs_list = []
        for raw in raw_list:
            epochs_list.append(self.__process_raw__(raw))

        features, labels = self.__create_dataset__(epochs_list)
        labels = labels[:, 2]  # drop index 0, 1 since they are not relevant
        features, labels = self.__filter_incorrect_labels__(features, labels)  # filter incorrect labels

        test_index = int(features.shape[0] * (1 - test_percent))
        train_x, train_y = features[:test_index], labels[:test_index]
        test_x, test_y = features[test_index:], labels[test_index:]

        if save:
            np.savez_compressed(file=output_filepath, train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)

        return (train_x, train_y), (test_x, test_y)

    def __create_dataset__(self, epochs_list: List[mne.Epochs]) -> (np.ndarray, np.ndarray):
        if len(epochs_list) == 0:
            self.logger.error('Error, epochs_list is empty')
            return

        epochs_list[0].load_data()
        features = epochs_list[0].get_data()
        labels = epochs_list[0].events

        for i in range(1, len(epochs_list)):
            epoch = epochs_list[i]
            epoch.load_data()

            sample_features = epoch.get_data()
            sample_labels = epoch.events

            if sample_features.shape[0] == 0:
                continue

            features = np.row_stack((features, sample_features))
            labels = np.row_stack((labels, sample_labels))

        return features, labels

    def __process_raw__(self, raw_file_name: str) -> mne.Epochs:
        raw = mne.io.read_raw_brainvision(raw_file_name, preload=True)

        if 'EOG' in raw.ch_names:
            raw.drop_channels('EOG')

        events, event_id = mne.events_from_annotations(raw)
        reject_criteria = {'eeg': 100e-6}

        return mne.Epochs(
            raw=raw, events=events, event_id=event_id,
            tmin=(-self.t_pre_stimulus), tmax=self.t_post_stimulus,
            baseline=(-.2, 0),
            reject=reject_criteria
        )

    def __get_vhdr_file_list__(self) -> List[str]:
        raw_list = []

        for folder in os.listdir(self.root_path):
            folder_path = os.path.join(self.root_path, folder)

            # skip files
            if os.path.isfile(folder_path):
                continue

            # create data path
            data_path = os.path.join(folder_path, 'Data')

            vhdr = self.__get_vhdr__(data_path)
            if vhdr is None:
                continue

            self.logger.debug(vhdr)
            raw_list.append(vhdr)

        return raw_list

    @staticmethod
    def __filter_incorrect_labels__(features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        label_index_list = []
        for i in range(0, labels.shape[0]):
            label = labels[i]
            if label > 9:
                label_index_list.append(i)

        # deletes incorrectly labeled indexes from both features and labels and returns copies of them
        return np.delete(features, label_index_list, axis=0), np.delete(labels, label_index_list, axis=0)

    @staticmethod
    def __get_vhdr__(folder_path: str) -> Union[str, None]:
        for file in os.listdir(folder_path):
            extension = os.path.splitext(file)[1]
            if extension == '.vhdr':
                return os.path.join(folder_path, file)

        return None


def main():
    preprocessing = Preprocessing()
    preprocessing.create_new_dataset()


if __name__ == '__main__':
    main()
