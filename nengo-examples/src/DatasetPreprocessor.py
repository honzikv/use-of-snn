# Wrapper for parsing data from p300 directory
import logging
import os
import random
from os import listdir
from typing import List, Tuple

import numpy as np

import mne
from mne.io.brainvision.brainvision import RawBrainVision

logger = logging.getLogger(__name__)


class DatasetPreprocessor:

    def __init__(self, dataset_path=os.path.join('..', 'datasets', 'p300')):
        self.dataset_path = dataset_path

    def parse_dataset(self, test_data_percentage: float = .25, seed=0) \
            -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        :param seed: seed for shuffling dataset
        :param test_data_percentage: percentage of training data instead used as test data - from (0, 0.5)
        :return: tuple (train_x, train_y, test_x, test_y)
        """

        if 0 > test_data_percentage > .5:
            test_data_percentage = .5

        all_epochs = []
        for folder in listdir(self.dataset_path):
            data_dir_path = os.path.join(self.dataset_path, folder, 'Data')
            experiment_epochs = DatasetPreprocessor.__parse_experiment__(data_dir_path)
            all_epochs.append(experiment_epochs)

        random.seed(seed)
        random.shuffle(all_epochs)
        features, labels = self.__concatenate_data__(all_epochs)

        test_data_index = features.shape[0] * (1 - test_data_percentage)
        train_x, train_y = features[:test_data_index], labels[:test_data_index]
        test_x, test_y = features[test_data_index:], labels[:test_data_index]

        return (train_x, train_y), (test_x, test_y)

    @staticmethod
    def __concatenate_data__(epochs: List[mne.Epochs]) -> Tuple[np.ndarray, np.ndarray]:
        epochs[0].load_data()
        features = np.ndarray(epochs[0].get_data())
        labels = np.ndarray(epochs[0].events)

        for i in range(1, len(epochs)):
            epoch = epochs[i]
            epoch.load_data()

            np.row_stack(features, epoch.get_data())
            np.row_stack(labels, epoch.events)

        return features, labels

    @staticmethod
    def __parse_experiment__(data_dir_path: str):
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
        epochs = mne.Epochs(raw, events, event_ids, tmin=-.4, tmax=.8)
        return epochs


data_parser = DatasetPreprocessor()
data_parser.parse_dataset(.23)
