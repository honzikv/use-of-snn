# Use of Spiking Neural Networks
A repository for my bachelor thesis 'Use of Spiking Neural Networks'. This repository contains several experiments with spiking networks on various datasets.

**Experiments:**
    
1. P300 Experiment - in folder "experiments/p300"
    * To start the simulation run p300_dataset_exp_convnet.ipynb
    * Visualization notebook - p300_stats_visualization.ipynb can be run to produce images present in the bachelor thesis
    
2. Training deep SNN using surrogate gradient - in folder "experiments/surr_grad_snn"
    * To start the simulation run Deep-snns-surrogate-grad.ipynb
    
3. BNCI Horizon experiment - in folder "experiments/bnci_horizon_dataset_exp"
    * Firstly, run the preprocessing to convert the dataset into usable format
    * Subsequently any of the four notebooks can be run to perform a part of the experiment:
        * For all samples run bnci_horizon_convnet_all_samples.ipynb for both CNN architectures and bnci_horzion_lstm_all_samples.ipynb for the LSTM network
        * For experiment on individuals run bnci_horizon_convnet_individuals.ipynb and bnci_horizon_lstm_individuals.ipynb


SpyTorch experiment also uses code from:
https://github.com/fzenke/spytorch

**References:**

Neftci, E.O., Mostafa, H., and Zenke, F. (2019). Surrogate Gradient Learning in Spiking Neural Networks: Bringing the Power of Gradient-based optimization to spiking neural networks. IEEE Signal Processing Magazine 36, 51–63. https://ieeexplore.ieee.org/document/8891809 preprint: https://arxiv.org/abs/1901.09948
