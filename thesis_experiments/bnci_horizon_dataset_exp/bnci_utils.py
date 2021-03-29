import os
import numpy as np
import tensorflow as tf
import keras
import nengo_dl
from tensorflow.python.keras import Input, Model
import nengo
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Conv2D, Dropout, AveragePooling2D, Flatten, Dense, BatchNormalization, Conv3D
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras import backend as K
import pandas as pd
from sklearn import metrics


# This file contains utility functions that are used across all notebooks that use BNCI dataset
# This is mainly done to reduce the overall boilerplate in the notebooks as the code to run the network would be mostly
# the same in all cases

def load_dataset(dataset_file_path):
    """
    Function for loading the dataset from npz numpy file
    :param dataset_file_path: path to the dataset file
    :return: loaded features and labels
    """
    dataset = np.load(dataset_file_path)
    return dataset['features'], dataset['labels']


def reshape_dataset(features, labels):
    """
    Reshapes the dataset to be usable within NengoDL. The features will be transformed for input shape (14, 360, 1)
    and the labels will be one-hot encoded
    :param features: numpy array containing features
    :param labels: numpy array containing labels
    :return: transformed features and labels
    """
    # Convert labels to one hot encoding
    labels = labels.reshape(-1, 1)
    labels = OneHotEncoder().fit_transform(labels).toarray()
    labels = labels.reshape((labels.shape[0], 1, -1))

    # Reshape features for the NN
    features = features.reshape((features.shape[0], 14, -1))  # reshape to channels x data
    features = features.reshape((features.shape[0], 1, -1))  # add time dimension

    return features, labels


def cnn_model(seed):
    inp = Input(shape=(14, 360, 1), name='input_layer')
    conv1 = Conv2D(filters=32, kernel_size=(5, 5), activation=tf.nn.relu, padding='same')(inp)
    dropout1 = Dropout(0.2, seed=seed)(conv1)
    avg_pool1 = AveragePooling2D(pool_size=(2, 2))(dropout1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu)(avg_pool1)
    dropout2 = Dropout(0.2, seed=seed)(conv2)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2))(dropout2)
    flatten = Flatten()(avg_pool2)
    dense1 = Dense(512, activation=tf.nn.relu)(flatten)
    dropout3 = Dropout(0.2, seed=seed)(dense1)
    dense2 = Dense(256, activation=tf.nn.relu)(dropout3)
    output = Dense(2, activation=tf.nn.softmax, name='output_layer')(dense2)
    return Model(inputs=inp, outputs=output)


def original_p300_model(seed=0):
    """
    Function to create the model from P300 dataset
    :return: Tensorflow model
    """
    inp = Input(shape=(14, 360, 1), name='input_layer')
    conv2d = Conv2D(filters=6, kernel_size=(3, 3), activation=tf.nn.relu)(inp)
    dropout1 = Dropout(0.5, seed=seed)(conv2d)
    avg_pooling = AveragePooling2D(pool_size=(1, 8), padding='same')(dropout1)
    flatten = Flatten()(avg_pooling)
    dense1 = Dense(100, activation=tf.nn.relu)(flatten)
    batch_norm = BatchNormalization()(dense1)
    dropout2 = Dropout(0.5, seed=seed)(batch_norm)
    output = Dense(2, activation=tf.nn.softmax, name='output_layer')(dropout2)

    return Model(inputs=inp, outputs=output)


def get_metrics(simulator, output_layer, x_test, y_test, minibatch_size, network_name):
    """
    Function for calculating metrics
    :param simulator: simulator instance
    :param input_layer: input layer reference
    :param output_layer: output layer reference
    :param x_test: features of the testing subset
    :param y_test: labels of the testing subset
    :param network_name: name of the network
    :return: accuracy, recall and precision metrics
    """

    # Truncate the remaining number of samples since the predict function does use minibatch
    samples = (x_test.shape[0] // minibatch_size) * minibatch_size
    x_test, y_test = x_test[:samples], y_test[:samples]

    predictions = simulator.predict(x_test)[output_layer]  # get result from output layer when predicting on x_test
    predictions = predictions[:, -1, :]  # get the last timestep
    predictions_argm = np.argmax(predictions, axis=-1)  # get predicted label

    y_test = np.squeeze(y_test, axis=1)  # remove time dimension
    y_test_argm = np.argmax(y_test, axis=-1)  # get labels

    precision = metrics.precision_score(y_true=y_test_argm, y_pred=predictions_argm,
                                        average='binary')  # get precision score
    recall = metrics.recall_score(y_true=y_test_argm, y_pred=predictions_argm, average='binary')  # get recall
    f1 = metrics.f1_score(y_true=y_test_argm, y_pred=predictions_argm, average='binary')  # get f1 score
    accuracy = metrics.accuracy_score(y_true=y_test_argm, y_pred=predictions_argm)  # get accuracy
    confusion_matrix = metrics.confusion_matrix(y_true=y_test_argm, y_pred=predictions_argm)  # get confusion matrix

    # Log the statistics
    print(f'{network_name}: accuracy = {accuracy * 100}%, precision = {precision}, '
          f'recall = {recall}, f1 = {f1}')
    print('Confusion matrix:')
    print(confusion_matrix)

    return accuracy, precision, recall, f1, confusion_matrix


def get_metrics_keras(model, x_test, y_test, network_name):
    predictions = model.predict(x_test)
    predictions = np.argmax(predictions, axis=-1)
    y_test = np.argmax(y_test, axis=-1)

    precision = metrics.precision_score(y_true=y_test, y_pred=predictions, average='binary')  # get precision score
    recall = metrics.recall_score(y_true=y_test, y_pred=predictions, average='binary')  # get recall
    f1 = metrics.f1_score(y_true=y_test, y_pred=predictions, average='binary')  # get f1 score
    accuracy = metrics.accuracy_score(y_true=y_test, y_pred=predictions)  # get accuracy
    confusion_matrix = metrics.confusion_matrix(y_true=y_test, y_pred=predictions)  # get confusion matrix

    # Log the statistics
    print(f'{network_name}: accuracy = {accuracy * 100}%, precision = {precision}, '
          f'recall = {recall}, f1 = {f1}')
    print('Confusion matrix:')
    print(confusion_matrix)

    return accuracy, precision, recall, f1, confusion_matrix


def run_ann(model, train, test, params_save_path, iteration, optimizer, loss, callbacks=None, valid=None,
            shuffle_training=True,
            batch_size=16,
            num_epochs=30):
    """
    Run analog network with cross-validation
    :param batch_size: batch size during training
    :param model: reference to the tensorflow model
    :param train: pair of training data (x_train, y_train)
    :param valid: pair of validation data (x_val, y_val)
    :param test: pair of testing data (x_test, y_test)
    :param params_save_path: output path to save weights of the network
    :param iteration: number of the iteration in CV
    :param shuffle_training: shuffle samples
    :param num_epochs: number of epochs to train for
    :return: accuracy, precision, recall, f1 and confusion matrix from the testing data
    """
    x_train, y_train = train[0], train[1]
    x_test, y_test = test[0], test[1]

    if valid is not None:
        x_valid, y_valid = valid[0], valid[1]

    converter = nengo_dl.Converter(model)

    with nengo_dl.Simulator(converter.net, minibatch_size=batch_size) as simulator:
        simulator.compile(optimizer=optimizer,
                          loss=loss,
                          metrics=['accuracy'])

        input_layer = converter.inputs[model.get_layer('input_layer')]  # get the input layer reference
        output_layer = converter.outputs[model.get_layer('output_layer')]  # get the output layer reference

        # fit the model with the training data
        simulator.fit(
            x={input_layer: x_train}, y={output_layer: y_train},
            validation_data=(
                {input_layer: x_valid}, {output_layer: y_valid}
            ) if valid is not None else None,
            epochs=num_epochs,
            shuffle=shuffle_training,
            callbacks=callbacks
            # early stop to avoid overfitting
        )

        simulator.save_params(params_save_path)  # save weights to the file

        # Get the statistics
        accuracy, precision, recall, f1, confusion_matrix = get_metrics(simulator, output_layer, x_test, y_test,
                                                                        batch_size,
                                                                        f'{iteration}. CNN')
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': confusion_matrix
        }


def run_snn(model, x_test, y_test, params_load_path, iteration, timesteps=50,
            scale_firing_rates=1000,
            synapse=0.01,
            batch_size=16):
    """
    Run model in spiking setting
    :param model: model reference
    :param x_test: testing features
    :param y_test: testing labels
    :param params_load_path: path to load parameters
    :param iteration: number of current iteration
    :param timesteps: number of timesteps
    :param scale_firing_rates: firing rate scaling
    :param synapse: synaptic smoothing
    :return: accuracy, precision, recall, f1 and confusion matrix from the testing data
    """
    converter = nengo_dl.Converter(
        model,
        swap_activations={tf.nn.relu: nengo.SpikingRectifiedLinear()},
        scale_firing_rates=scale_firing_rates,
        synapse=synapse
    )  # create a Nengo converter object and swap all relu activations with spiking relu

    with converter.net:
        nengo_dl.configure_settings(stateful=False)

    output_layer = converter.outputs[model.get_layer('output_layer')]  # output layer for simulator

    x_test_tiled = np.tile(x_test, (1, timesteps, 1))  # tile test data to timesteps

    with nengo_dl.Simulator(converter.net) as simulator:
        simulator.load_params(params_load_path)

        # Get the statistics
        accuracy, precision, recall, f1, confusion_matrix = get_metrics(simulator, output_layer, x_test_tiled, y_test,
                                                                        batch_size,
                                                                        f'{iteration}. CNN (SNN conversion)')
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': confusion_matrix
        }


def create_data_df(ann, snn, num_iterations):
    """
    Function takes in ann and snn list with dictionaries containing metrics in each iteration
    and maps them to dictionary that can be used in pandas
    :param ann: list of ANN results from function run_ann
    :param snn: list of SNN results from function run_snn
    :param num_iterations: the number of iterations
    :return: pandas dataframe with statistics from each iteration for SNN and ANN
    """

    return pd.DataFrame({
        'iterations': [x for x in range(1, num_iterations + 1)],
        'ann_accuracy': [x['accuracy'] for x in ann],
        'ann_precision': [x['precision'] for x in ann],
        'ann_recall': [x['recall'] for x in ann],
        'ann_f1': [x['f1'] for x in ann],
        'snn_accuracy': [x['accuracy'] for x in snn],
        'snn_precision': [x['precision'] for x in snn],
        'snn_recall': [x['recall'] for x in snn],
        'snn_f1': [x['f1'] for x in snn]
    })


def create_stats_df(df: pd.DataFrame):
    """
    Function takes in
    :param df: dataframe from create_data_df function
    :return: pandas dataframe with
    """
    data_stats = {
        'models': ['ann', 'snn'],
        'average_accuracy': [],
        'max_accuracy': [],
        'accuracy_std': [],
        'average_precision': [],
        'max_precision': [],
        'average_recall': [],
        'max_recall': [],
        'average_f1': [],
        'max_f1': []
    }

    # slightly less code if we iterate over snn_{metric_name} in dictionary
    for model in ['ann', 'snn']:
        data_stats['average_accuracy'].append(df[f'{model}_accuracy'].mean())
        data_stats['accuracy_std'].append(df[f'{model}_accuracy'].std())
        data_stats['average_precision'].append(df[f'{model}_precision'].mean())
        data_stats['average_recall'].append(df[f'{model}_recall'].mean())
        data_stats['average_f1'].append(df[f'{model}_f1'].mean())
        data_stats['max_accuracy'].append(df[f'{model}_accuracy'].max())
        data_stats['max_f1'].append(df[f'{model}_f1'].max())
        data_stats['max_precision'].append(df[f'{model}_precision'].max())
        data_stats['max_recall'].append(df[f'{model}_recall'].max())

    return pd.DataFrame(data_stats)


def print_confusion_matrices(ann, snn=None):
    """
    Prints confusion matrix in each iteraiton
    :param ann: list of results for ANN model from run_ann function
    :param snn: list of results for SNN model from run_snn function
    """

    # Print confusion matrices for ANN
    conf_matrices_ann = [x['confusion_matrix'] for x in ann]
    print('Confusion matrices for the ANN:')
    for confusion_matrix in conf_matrices_ann:
        print(confusion_matrix, '\n')

    if snn is None:
        return

    # Print confusion matrices for SNN
    conf_matrices_snn = [x['confusion_matrix'] for x in snn]
    print('Confusion matrices for the SNN')
    for confusion_matrix in conf_matrices_snn:
        print(confusion_matrix, '\n')
