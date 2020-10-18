# Wrapper for parsing data from p300 directory
import logging
import os
from os import listdir

import mne
from mne.io.brainvision.brainvision import RawBrainVision

logger = logging.getLogger(__name__)


class DataParser:
    _required_file_extensions = ['.eeg', '.vhdr', '.vmrk']

    def __init__(self, dataset_path=os.path.join('..', 'datasets', 'p300')):
        self.dataset_path = dataset_path

    def parse_dataset(self):
        pass

    @staticmethod
    def visualize_data(raw_brain_vision: RawBrainVision):
        events = mne.find_events(raw_brain_vision)


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

        raw_brain_vision = mne.io.read_raw_brainvision(
            vhdr_fname=vhdr_file,
            eog=['Fz', 'Cz', 'Pz', 'EOG']
        )
        raw_brain_vision.plot()
        return raw_brain_vision


data_parser = DataParser()
data_parser.__parse_experiment__(
    os.path.join(data_parser.dataset_path, 'Experiment_341_P3_Numbers', 'Data')
)
