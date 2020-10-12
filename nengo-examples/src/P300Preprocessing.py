import os
import zipfile

from .DataParser import DataParser

class P300Preprocessing:

    def __init__(self, zip_path: str = os.path.join('..', 'datasets', 'PROJECT_DAYS_NUMBERS.zip')
                 , working_directory: str = os.path.join('..', 'datasets', 'p300')):
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
            dataParser = DataParser()
            for file in os.listdir(os.path.join(self.working_directory, folder, 'Data')):


preprocessing = P300Preprocessing()
preprocessing.unzip()
