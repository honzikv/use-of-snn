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
    :return:
    """
    dataset = np.load(dataset_file_path)
    return dataset['features'], dataset['labels']




def cnn_model(seed=0):
    """
    Function to create model from BNCI dataset
    :return: Tensorflow model
    """
    inp = Input(shape=(14, 360, 1), name='input_layer')
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.relu)(inp)
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
    inp = Input(shape=(3, 1200, 1), name='input_layer')
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
    f1 = metrics.f1_score(y_true=y_test_argm, y_pred=predictions_argm, average='binary')
    accuracy = metrics.accuracy_score(y_true=y_test_argm, y_pred=predictions_argm)  # get accuracy
    confusion_matrix = metrics.confusion_matrix(y_true=y_test_argm, y_pred=predictions_argm)

    # Log the statistics
    print(f'{network_name}: accuracy = {accuracy * 100}%, precision = {precision}, '
          f'recall = {recall}, f1 = {f1}')
    print('Confusion matrix:')
    print(confusion_matrix)

    return accuracy, precision, recall, f1, confusion_matrix


def run_ann(model, train, test, params_save_path, iteration, epochs=30, val=None, shuffle_training=True,
            use_early_stopping_callback=True,
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.BinaryCrossentropy()):
    """
    Run ann via Nengo simulator. This fits the given model with the training data (train)
    and validates it using validation data (valid). Then accuracy is calculated using the test data (test)
    and weights are saved to params_save_path
    :param loss: loss function for training
    :param optimizer: optimizer for training
    :param use_early_stopping_callback: whether to use early stopping callback or not
    :param shuffle_training: whether to shuffle data (default true)
    :param model: tensorflow model created from create_model() function
    :param train: pair of features and labels from training data
    :param val: pair of features and labels from validation data
    :param test: pair of features and labels from test data
    :param params_save_path: output path to save weights of the network for SNN testing
    :return accuracy, precision, recall, f1 and confusion matrix from the testing data
    """

    # unwrap into training and testing data for each subset
    x_train, y_train = train[0], train[1]

    if val is not None:
        x_val, y_val = val[0], val[1]

    x_test, y_test = test[0], test[1]

    converter = nengo_dl.Converter(model)
    with nengo_dl.Simulator(converter.net, minibatch_size=64) as simulator:
        # Compile the model with binary cross-entropy and Adam optimizer
        simulator.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )

        input_layer = converter.inputs[model.get_layer('input_layer')]  # get nengo input layer
        output_layer = converter.outputs[model.get_layer('output_layer')]  # get nengo output layer

        # Train the model, if no validation data is provided use early stopping callback (if enabled) on training data
        # instead
        if val is None:
            # noinspection PyTypeChecker
            simulator.fit(
                x={input_layer: x_train}, y={output_layer: y_train},
                epochs=epochs,
                shuffle=shuffle_training,
                callbacks=[EarlyStopping(patience=5, verbose=1, restore_best_weights=True,
                                         monitor='probe_loss')] if use_early_stopping_callback else None
            )
        else:
            # noinspection PyTypeChecker
            simulator.fit(
                x={input_layer: x_train}, y={output_layer: y_train},
                validation_data=({input_layer: x_val}, {output_layer: y_val}),
                epochs=30,
                shuffle=shuffle_training,
                callbacks=[EarlyStopping(patience=5, verbose=1, restore_best_weights=True)]
            )

        simulator.save_params(params_save_path)  # save params for SNN

        # Get all metrics
        accuracy, precision, recall, f1, confusion_matrix = get_metrics(simulator, output_layer, x_test, y_test,
                                                                        minibatch_size=simulator.minibatch_size,
                                                                        network_name=f'{iteration}. ANN')
        # Return metrics in dictionary
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': confusion_matrix
        }


def run_snn(model, test, params_load_path, timesteps, scale_firing_rates, synapse, iteration,
            swap_activations={tf.nn.relu: nengo.SpikingRectifiedLinear()}):
    """
    Runs SNN on test data. Loads pre-trained weights from params_load path and uses timesteps, scale_firing_rates and synapse
    parameters for simulator.
    :param model: reference to the tensorflow model
    :param test: reference to the test features and labels
    :param params_load_path: path to the saved weights of the ANN
    :param timesteps: number of timesteps - i.e. how long is the input streamed to the network
    :param scale_firing_rates: firing rate scaling - amplifies spikes
    :param synapse: synaptic smoothing
    :param iteration: iteration to print the result
    :param swap_activations: dictionary containing activation swaps when converting to the spiking network
    :return: accuracy, precision, recall, f1 and confusion matrix from the testing data
    """

    # Conversion of the TensorFlow model to a spiking Nengo model
    converter = nengo_dl.Converter(
        model=model,
        swap_activations=swap_activations,
        scale_firing_rates=scale_firing_rates,
        synapse=synapse
    )

    x_test, y_test = test[0], test[1]  # get test features and labels

    with converter.net:
        nengo_dl.configure_settings(stateful=False)

    output_layer = converter.outputs[model.get_layer('output_layer')]  # output layer for simulator
    x_test_time_tiled = np.tile(x_test, (1, timesteps, 1))  # tile x_test to match desired timesteps for simulator

    with nengo_dl.Simulator(converter.net, minibatch_size=41, progress_bar=False) as simulator:
        simulator.load_params(params_load_path)

        # Name of the network for print in get_metrics function
        name = f'{iteration}. SNN [timesteps={timesteps}, scale_firing_rates={scale_firing_rates}, synapse={synapse}]'
        accuracy, precision, recall, f1, confusion_matrix = get_metrics(simulator, output_layer, x_test_time_tiled,
                                                                        y_test,
                                                                        minibatch_size=simulator.minibatch_size,
                                                                        network_name=name)
        # Return metrics in dictionary
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': confusion_matrix
        }
