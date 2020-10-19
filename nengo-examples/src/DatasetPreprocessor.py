# Wrapper for parsing data from p300 directory
import logging
import os
from os import listdir
import numpy as np

import mne
from mne.io.brainvision.brainvision import RawBrainVision

logger = logging.getLogger(__name__)


class DatasetPreprocessor:

    def __init__(self, dataset_path=os.path.join('..', 'datasets', 'p300')):
        self.dataset_path = dataset_path

    def parse_dataset(self, test_data_percentage: float = .25):
        """
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

        for epoch in all_epochs:
            print(epoch.get_data().shape)


    @staticmethod
    def __parse_experiment__(data_dir_path: str):
        vhdr_file = None

        for file in listdir(data_dir_path):
            extension = os.path.splitext(file)[1]
            if extension == '.vhdr':
                vhdr_file = os.path.join(data_dir_path, file)

        if vhdr_file is None:
            logger.log(logging.ERROR,
                       'VHDR file not found, skipping this experiment, it is probably corrupt or missing data')
            return

        raw: RawBrainVision = mne.io.read_raw_brainvision(vhdr_fname=vhdr_file)
        raw.drop_channels('EOG')

        events, event_ids = mne.events_from_annotations(raw)
        epochs = mne.Epochs(raw, events, event_ids, tmin=-.4, tmax=.8)
        return epochs


data_parser = DatasetPreprocessor()
data_parser.__parse_experiment__(
    os.path.join(data_parser.dataset_path, 'Experiment_341_P3_Numbers', 'Data')
)
