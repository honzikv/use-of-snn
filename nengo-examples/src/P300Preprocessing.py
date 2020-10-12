import os
import zipfile

from .DataParser import DataParser


class P300Preprocessing:

    def __init__(self, working_directory: str = os.path.join('..', 'datasets', 'p300')):
        self.working_directory = os.path.abspath(working_directory)

    def get_dataset(self):
        self.unzip()
        self.preprocess_data()

    def unzip(self):
        for folder in os.listdir(self.working_directory):
            self.extract_data_zips(os.path.join(self.working_directory, folder))

    def extract_data_zips(self, folder):
        data_path = os.path.join(folder, 'Data')
        for item in os.listdir(data_path):
            if item.endswith('zip'):
                with zipfile.ZipFile(os.path.join(self.working_directory, data_path, item)) as zip_file:
                    zip_file.extractall(os.path.join(self.working_directory, data_path))

    def preprocess_data(self):
        for folder in os.listdir(self.working_directory):
            files = {}
            data_path = os.path.join(self.working_directory, folder, 'Data')
            for file in os.listdir(data_path):
                extension = os.path.splitext(file)[1]
                files[extension] = os.path.join(data_path, file)

            try:
                data_parser = DataParser(eeg_file_path=files['.eeg'],
                                         vhdr_path=files['.vhdr'],
                                         vmrk_path=files['.vmrk'])



            except KeyError:
                print('Incomplete data, skipping')
                return


preprocessing = P300Preprocessing()
preprocessing.unzip()
