import os

import keras
import nengo
import tensorflow as tf
import nengo_dl
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split, ShuffleSplit
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, Dropout, AveragePooling2D, Flatten, Dense

os.makedirs('nengo', exist_ok=True)

dataset_path = os.path.join('..', 'datasets', 'output', 'VarekaGTNEpochs.mat')
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)


def get_dataset(file: str):
    np_file = loadmat(file)
    target_data = np_file['allTargetData']
    non_target_data = np_file['allNonTargetData']

    features = np.concatenate((target_data, non_target_data))
    target_labels = np.tile(np.array([1, 0]), (target_data.shape[0], 1))
    non_target_labels = np.tile(np.array([0, 1]), (non_target_data.shape[0], 1))
    labels = np.vstack((target_labels, non_target_labels))

    threshold = 100.0
    result_x, result_y = [], []
    for i in range(features.shape[0]):
        if not np.max(np.abs(features[i])) > threshold:
            result_x.append(features[i])
            result_y.append(labels[i])

    features, labels = np.array(result_x), np.array(result_y)
    features = features.reshape((features.shape[0], 1, -1))
    labels = labels.reshape((labels.shape[0], 1, -1))
    print(features.shape)
    print(labels.shape)
    print(labels)

    print('parsed features and labels, features: {}, labels: {}'.format(features.shape, labels.shape))
    return features, labels


def create_model():
    # model = Sequential()
    # model.add(Conv2D(filters=6, kernel_size=(3, 3), activation=keras.activations.elu, input_shape=(3, 1200, 1),
    #                  name='input_layer'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5, seed=0))
    # model.add(AveragePooling2D(pool_size=(1, 8)))
    # model.add(Flatten())
    # model.add(Dense(100, activation=keras.activations.elu))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5, seed=0))
    # model.add(Dense(2, activation=keras.activations.softmax, name='output_layer'))
    # model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

    input = Input(shape=(1200, 3, 1), name='input_layer')
    conv2d = Conv2D(filters=6, kernel_size=(3, 3), activation=keras.activations.elu)(input)
    dropout1 = Dropout(0.5, seed=0)(conv2d)
    avg_pooling = AveragePooling2D(pool_size=(1, 8), padding='same')(dropout1)
    flatten = Flatten()(avg_pooling)
    dense1 = Dense(100, activation=keras.activations.elu)(flatten)
    batch_norm = BatchNormalization()(dense1)
    dropout2 = Dropout(0.5, seed=0)(batch_norm)
    output = Dense(2, activation=keras.activations.softmax, name='output_layer')(dropout2)

    return Model(inputs=input, outputs=output)


def run_nengo_ann(train_x, train_y, valid_x, valid_y, test_x, test_y,
                  param_output=os.path.join('nengo', 'network_params')):
    model = create_model()
    converter = nengo_dl.Converter(model)

    early_stop = EarlyStopping(patience=5, verbose=1, restore_best_weights=True)

    # run ann with nengo
    with nengo_dl.Simulator(converter.net, minibatch_size=16) as simulator:
        simulator.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        print('x: {}, y: {}'.format(train_x.shape, train_y.shape))
        print('vx: {}, vy: {}'.format(len(valid_x.shape), len(valid_y.shape)))
        history = simulator.fit(x=train_x, y=train_y,
                                validation_data=(valid_x, valid_y),
                                epochs=30,
                                shuffle=True,  # delete later to see if it makes any difference
                                callbacks=[early_stop]
                                )

        ann_results = simulator.evaluate(x=test_x, y=test_y)
        simulator.save_params(param_output)

    return model, history, ann_results


def run_nengo_spiking(test_x, test_y, model, params_path=os.path.join('nengo', 'network_params')):
    converter = nengo_dl.Converter(model, swap_activations={tf.nn.elu: nengo.SpikingRectifiedLinear()},
                                   scale_firing_rates=1000, synapse=0.01)

    timesteps = 30
    test_x = np.tile(test_x, (1, timesteps, 1))

    with converter.net:
        nengo_dl.configure_settings(stateful=False)

    nengo_input = converter.model.get_layer('input_layer')
    nengo_output = converter.model.get_layer('output_layer')

    with nengo_dl.Simulator(converter.net, minibatch_size=10, progress_bar=False) as simulator:
        simulator.load_params(params_path)
        data = simulator.predict({nengo_input: test_x})
        predictions = np.argmax(data[nengo_output][:, -1], axis=-1)
        accuracy = (predictions == test_y).mean()
        print('accuracy: {}%'.format(accuracy * 100))

    return accuracy


features, labels = get_dataset(dataset_path)
train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.25, random_state=seed, shuffle=True)

print('total train_x data: {}, total test_x data: {}, total train_y data: {}, total test_y data: {}'
      .format(train_x.shape, test_x.shape, train_y.shape, test_y.shape))

valid, test, snn = [], [], []

counter = 1
monte_carlo = ShuffleSplit(n_splits=30, test_size=0.25, random_state=seed)
for train, validation in monte_carlo.split(train_x):
    print('iteration {}'.format(counter))
    counter += 1

    print(train, validation)
    curr_train_x, curr_train_y = train_x[train], train_y[train]
    curr_valid_x, curr_valid_y = train_x[validation], train_y[validation]

    print(
        'current training data x: {}, current training data y: {}, current validation data x: {}, '
        'current validation data y: {}'
            .format(curr_train_x.shape, curr_train_y.shape, curr_valid_x.shape, curr_valid_y.shape)
    )

    model, history, results = run_nengo_ann(
        train_x=curr_train_x, train_y=curr_train_y, valid_x=curr_valid_x, valid_y=curr_valid_y,
        test_x=test_x, test_y=test_y
    )

    valid.append(history)
    test.append(results)
    print('accuracy of ann model: {}%'.format(results['probe_accuracy'] * 100))

    spiking_acc = run_nengo_spiking(test_x=test_x, test_y=test_y, model=model)
    snn.append(spiking_acc)

