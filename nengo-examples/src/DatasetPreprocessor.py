# Wrapper for parsing data from p300 directory
import logging
import os
import random
from os import listdir
from typing import List

import numpy as np

import mne
from mne.io.brainvision.brainvision import RawBrainVision

logger = logging.getLogger()


class DatasetPreprocessor:

    def __init__(self,
                 dataset_path=os.path.join('..', 'datasets', 'p300'),
                 log_info=False,
                 export=True,
                 output_path=os.path.join('..', 'datasets', 'p300'),
                 output_filename='p300_dataset',
                 t_min=-200,
                 t_end=1000
                 ):
        self.dataset_path = dataset_path
        self.export = export
        self.output_path = output_path
        self.log_info = log_info
        self.output_filename = output_filename
        self.t_min = t_min
        self.t_end = t_end

        if not log_info:
            mne.set_log_level(verbose='ERROR')

    def load_from_file(self, file_path):
        pass

    def create_dataset(self, test_data_percentage: float = .25, seed=0):
        """
        :param seed: seed for dataset shuffle to provide consistent results
        :param test_data_percentage: percentage of training data instead used as test data - from (0, 0.5)
        :return: tuple (train_x, train_y, test_x, test_y)
        """

        if 0 > test_data_percentage > .5:
            test_data_percentage = .25

        all_epochs = []
        for folder in listdir(self.dataset_path):
            path = os.path.join(self.dataset_path, folder)
            if os.path.isfile(path):
                continue

            data_dir_path = os.path.join(path, 'Data')
            experiment_epochs = self.__parse_experiment__(data_dir_path)
            all_epochs.append(experiment_epochs)

        random.seed(seed)
        random.shuffle(all_epochs)
        logger.info('concatenating loaded data')
        features, labels = self.__concatenate_data__(all_epochs)

        logger.info('splitting to train and test data')
        test_data_index = int(features.shape[0] * (1 - test_data_percentage))
        train_x, train_y = features[:test_data_index], labels[:test_data_index]
        test_x, test_y = features[test_data_index:], labels[test_data_index:]

        if self.export:
            save_filename = os.path.join(self.output_path, self.output_filename)
            np.savez_compressed(file=save_filename, train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
            logger.info('Output dataset saved to:', save_filename)

        return train_x, train_y, test_x, test_y

    @staticmethod
    def __concatenate_data__(epochs: List[mne.Epochs]):
        if len(epochs) <= 1:
            logger.error('Error while parsing epochs')

        epochs[0].load_data()
        features = epochs[0].get_data()
        labels = list(epochs[0].events[:, 2])  # only class is relevant

        for i in range(1, len(epochs)):
            epoch = epochs[i]
            epoch.load_data()

            features = np.row_stack((features, epoch.get_data()))
            epoch_events = list(epoch.events[:, 2])
            labels += epoch_events

        return features, np.array(labels)

    def __parse_experiment__(self, data_dir_path: str):
        vhdr_file = None

        for file in listdir(data_dir_path):
            extension = os.path.splitext(file)[1]
            if extension == '.vhdr':
                vhdr_file = os.path.join(data_dir_path, file)

        # log that there is no vhdr file in folder and return
        if vhdr_file is None:
            logger.log(logging.ERROR,
                       'VHDR file not found, skipping this folder...')
            return

        raw: RawBrainVision = mne.io.read_raw_brainvision(vhdr_fname=vhdr_file)

        if 'EOG' in raw.ch_names:
            raw.drop_channels('EOG')

        events, event_ids = mne.events_from_annotations(raw)

        reject_criteria = {'eeg': 100e-6}
        epochs = mne.Epochs(raw=raw, events=events, event_id=event_ids,
                            tmin=self.t_min, tmax=self.t_end,
                            baseline=(-200, 0),
                            reject=reject_criteria
                            )

        return epochs


data_parser = DatasetPreprocessor()
train_x, train_y, test_x, test_y = data_parser.create_dataset(.23, 1)
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)
