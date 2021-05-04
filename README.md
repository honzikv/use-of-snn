# Use of Spiking Neural Networks
This repository contains all three experiments that were conducted in my thesis 'Use of Spiking Neural Networks'. All three experiments are reproducable using combination of Python scripts and Jupyter Notebooks.

# Setting up the environment
Firstly, it is recommended to set up a new Python/Conda virtual environment so that there are no conflicts in dependencies. The root folder contains **requirements.txt** file that can be used to download all dependencies except PyTorch and CUDA support (there were some issues when installing them via pip).

This quick setup guide only shows how to get the repo working using Anaconda (default Python can be used as well, however, Anaconda has a much easier setup and environment management).

1. Download the repository to your computer (Windows 10 is recommended, however, all files should be compatible with macOS or Linux as well)
2. Download and install Anaconda (https://www.anaconda.com/), if you have not already, make sure that the command "conda" is saved in the PATH variable (for Windows) as it will be necessary to set up the environment
3. Open command line - "Command Prompt" on Windows or Terminal on Linux (ideally in the same folder as the downloaded repository) and create a new environment like so: `conda create -n name_of_the_environment python=3.8.5`
	- This will create an empty environment with Python 3.8.5
	- Be sure to change the name to the actual name of the environment
4. Now activate the environment using `conda activate name_of_the_environment`
5. Install dependencies from the root folder of the repository using `pip install -r requirements.txt`
6. Now PyTorch can be installed as well using `conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge` if you have Nvidia GPU available (with CUDA cores) or `conda install pytorch torchvision torchaudio cpuonly -c pytorch` if you do not have GPU or wish to only use CPU
7. In some cases, PyTorch might install a newer version of NumPy which may be incompatible with TensorFlow. To fix this, you can roll back to the previous version by running: `pip install numpy==1.9.5`

# Running the experiments
Jupyter notebooks (files that end with *.ipynb* extension) are required to run all three experiments (Jupyter is already present in the requirements.txt files). To run any notebook, Jupyter server needs to be started. This serves as an editor to run and edit any code. Open jupyter server (ideally again in the root folder of the downloaded repository) using: `jupyter notebook`

This should either open a browser or show a prompt. Alternatively, an URL should appear in the console, which redirects to the web application. The notebooks can also be run in other IDEs such as PyCharm Professional or Visual Studio Code.

The experiments are located in the **experiments** folder:

 - **bnci_horizon_experiment** - uses BNCI Horizon Spatial attention shifts to colored items dataset
 - **p300_experiment** - uses a P300 multi-subject dataset
 - **surrogate_gradient_experiment** - uses surrogate gradient to train spiking networks on MNIST and Fashion MNIST

## BNCI Horizon Experiment 
To perform this experiment, the data from each participant need to be downloaded first. Go to: http://bnci-horizon-2020.eu/database/data-sets and download all 18 files from the 28th experiment named "Spatial attention shifts to colored items - an EEG-based BCI (002-2020)". 

 - Place the downloaded files in the **experiments/bnci_horizon_experiment/dataset** folder
 - This folder should only contain the **P01 - P18.mat** files and the **__init\_\_.py** file
 - Renaming the files will require changes in all the notebooks as well

Note that the P01 - P18.mat files need to be preprocessed, either by the **preprocessing.py** script or directly in a notebook in **preprocessing.ipynb** file. After running it, a new folder named **dataset_result** should create - this folder contains all preprocessed data.

Now any of the four notebooks can be run, here is a brief description what each one does:

 - **bnci\_horizon\_convnet\_all\_samples.ipynb** - is used to run the entire dataset, samples from the female subjects and samples from the male subjects on the two used CNN models
 - **bnci\_horizon\_convnet\_individuals.ipynb** - is used to run both CNN models on data from a single subject
 - **bnci\_horizon\_lstm\_all\_samples.ipynb** - runs the LSTM model on the entire dataset, samples from the female subjects and samples from the male subjects
 - **bnci\_horizon\_lstm\_individuals.ipynb** - runs the LSTM model on data from a single subject

The experiment contains three different models of neural networks - two CNNs and one LSTM. Only the CNNs were convertible, and therefore the LSTM model has separate scripts as it is run only in TensorFlow. 

Note that to run any of the files, **bnci_utils.py** needs to be in the same folder as it contains functionality that is shared across all four notebooks.

## Multi-subject P300 dataset experiment
This experiment uses EEG samples from a P300 dataset that can be downloaded here: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/G9RRLN. Place the downloaded file in the **experiments/p300_experiment/dataset** folder and do not rename it.

Here, one notebook contains code to run the experiment, while the other can be used for visualization (though it is not necessary). To run the experiment open the **p300_dataset_exp_convnet.ipynb** file and execute all cells. The output of the dataset can then be used in the **p300_stats_visualization.ipynb** file.

## Surrogate gradient experiment
This experiment does not require any additional downloads. To run it, simply execute all cells in the **surrogate_gradient_deep_snns.ipynb** notebook. Note that it uses PyTorch and the simulation is relatively long (compared to the other two experiments) - thus using GPU is highly recommended (with GTX 1060 6GB one model took around 40 - 60 mins).

Part of the notebook uses code from https://github.com/fzenke/spytorch.
