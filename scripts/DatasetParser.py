import os
from typing import Union, Tuple

import mne
import numpy as np


class DatasetParser:

    def __init__(self,
                 dataset_root_path=os.path.join('..', 'datasets', 'p300'),
                 log_mne=False,
                 t_pre_stimulus_s=0.2,
                 t_post_stimulus_s=1.0
                 ):
        self.dataset_root_path = dataset_root_path
        self.log_mne = log_mne
        self.t_pre_stimulus_s = t_pre_stimulus_s
        self.t_post_stimulus_s = t_post_stimulus_s

    def parse_from_root(self):
        file_list = self.__get_file_list__()

        data = []  # all epochs
        for item in file_list:
            raw_file_path = item[0]
            target_file_path = item[1]

            target = self.__get_target__(target_file_path)
            epochs = self.__process_raw__(raw_file_path)

            data.append({
                'target': target,
                'epochs': epochs
            })

        return self.__create_dataset__(data)

    @staticmethod
    def save_to_file(target_features: np.ndarray, target_labels: np.ndarray, non_target_features: np.ndarray,
                     non_target_labels: np.ndarray,
                     output_file=os.path.join('..', 'datasets', 'output', 'p300-target-nontarget')):

        os.makedirs(exist_ok=True, name=os.path.dirname(output_file))

        np.savez_compressed(output_file,
                            target_features=target_features, target_labels=target_labels,
                            non_target_features=non_target_features, non_target_labels=non_target_labels)

    def __get_file_list__(self):
        file_list = []

        for folder in os.listdir(self.dataset_root_path):
            curr_folder_path = os.path.join(self.dataset_root_path, folder)

            if os.path.isfile(curr_folder_path):
                continue

            data_path = os.path.join(curr_folder_path, 'Data')
            required_files = self.__get_required_files__(data_path)
            if required_files is not None:
                file_list.append(required_files)

        return file_list

    @staticmethod
    def __get_required_files__(data_path) -> Union[Tuple[str, str], None]:
        vhdr = None
        target = None

        for file in os.listdir(data_path):
            extension = os.path.splitext(file)[1]
            if extension == '.vhdr':
                vhdr = os.path.join(data_path, file)
            elif extension == '.txt':
                target = os.path.join(data_path, file)

        if vhdr is not None and target is not None:
            return vhdr, target

        return None

    def __process_raw__(self, raw_file_path):
        raw = mne.io.read_raw_brainvision(raw_file_path, preload=True)

        if 'EOG' in raw.ch_names:
            raw.drop_channels('EOG')

        events, event_id = mne.events_from_annotations(raw)
        reject_criteria = {'eeg': 100e-6}

        return mne.Epochs(
            raw=raw, events=events, event_id=event_id,
            tmin=-self.t_pre_stimulus_s, tmax=self.t_post_stimulus_s,
            baseline=(-.2, 0),
            reject=reject_criteria,
            preload=True
        )

    @staticmethod
    def __get_target__(target_file_path):
        with open(target_file_path, mode='r') as file:
            for line in file:
                if line.startswith('the number thought'):
                    tokens = line.split(':')
                    num = tokens[1].strip()
                    return int(num) if num.isnumeric() else None

        return None

    @staticmethod
    def __create_dataset__(data) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        number_index = 2
        target_features, target_labels, non_target_features, non_target_labels = [], [], [], []
        for item in data:
            epochs = item['epochs']
            target = item['target']

            # iterate over each epoch in experiment

            for i in range(0, epochs.events.shape[0]):
                # filter errors
                if epochs.events[i][number_index] >= 9 or epochs.events[i][number_index] < 0:
                    continue

                if target is not None and epochs.events[i][number_index] == target:
                    target_features.append(epochs.get_data()[i])
                    target_labels.append(target)
                else:
                    non_target_features.append(epochs.get_data()[i])
                    non_target_labels.append(epochs.events[i][number_index])

        # finally, transform to np array
        target_features = np.array(target_features)
        target_labels = np.array(target_labels)
        non_target_features = np.array(non_target_features)
        non_target_labels = np.array(non_target_labels)

        return target_features, target_labels, non_target_features, non_target_labels


dataset_parser = DatasetParser()
target_features, target_labels, non_target_features, non_target_labels = dataset_parser.parse_from_root()
dataset_parser.save_to_file(target_features, target_labels, non_target_features, non_target_labels)
print(target_features.shape, target_labels.shape, non_target_features.shape, non_target_labels.shape)
