class DataParser:

    def __init__(self, eeg_file_path: str, vhdr_path: str, vmrk_path: str):
        self.eeg_file_path = eeg_file_path
        self.vhdr_path = vhdr_path
        self.vmrk_path = vmrk_path

    def parse(self):
        pass


import tensorflow as tf
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
