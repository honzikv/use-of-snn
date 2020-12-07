from typing import List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch
import os
import torchvision
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.float


def convert_to_firing_time(x, tau=20, thr=0.2, tmax=1.0, epsilon=1e-7):
    idx = x < thr
    x = np.clip(x, thr + epsilon, 1e9)
    T = tau * np.log(x / (x - thr))
    T[idx] = tmax
    return T


def sparse_data_generator(x, y, batch_size, num_steps, num_units, time_step=1e-3, shuffle=True):
    labels = np.array(y, dtype=np.int)
    number_of_batches = len(x) // batch_size
    sample_index = np.arange(len(x))

    tau_eff = 20e-3 / time_step
    firing_times = np.array(convert_to_firing_time(x, tau_eff, tmax=num_steps), dtype=np.int)
    unit_numbers = np.arange(num_units)

    # shuffle for training
    if shuffle:
        np.random.shuffle(sample_index)

    curr_batch = 0
    while curr_batch < number_of_batches:
        batch_idx = sample_index[batch_size * curr_batch: batch_size * (curr_batch + 1)]
        coo = [[], [], []]

        for bc, idx in enumerate(batch_idx):
            c = firing_times[idx] < num_steps
            times, units = firing_times[idx][c], unit_numbers[c]

            batch = [bc for _ in range(len(times))]
            coo[0].extend(batch)
            coo[1].extend(times)
            coo[2].extend(units)

        i = torch.LongTensor(coo).to(device)
        v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)

        x_batch = torch.sparse.FloatTensor(i, v, torch.Size([batch_size, num_steps, num_units])).to(device)
        y_batch = torch.tensor(labels[batch_idx], device=device, dtype=torch.long)

        yield x_batch.to(device=device), y_batch.to(device=device)

        curr_batch += 1


class SurrGradSpike(torch.autograd.Function):
    scale = 100.0  # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (SurrGradSpike.scale * torch.abs(input) + 1.0) ** 2
        return grad


class DeepSNNModel:

    def __init__(self, units: List[int], scale_params: Tuple = (1.0, 1.0), recurrent=False, time_step=1e-3,
                 tau_mem=10e-3,
                 tau_syn=5e-3):
        """
        :param units: list of units
        :param scale_params: tuple of one or two values
        :param recurrent: whether the network is recurrent one - additional recurrent weights are used
        :param time_step: duration of timestep in seconds
        :param tau_mem: membrane tau parameter
        :param tau_syn: synapse tau parameter
        """
        self.units = units
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.alpha = float(np.exp(-time_step / tau_syn))
        self.beta = float(np.exp(-time_step / tau_mem))

        if (len(scale_params) == 2):
            self.weight_scale = scale_params[0] - (scale_params[1] - self.beta)
        else:
            self.weight_scale = scale_params[0]
        self.weights = self.init_weights()
        self.recurrent_weights = False if not recurrent else self.init_recurrent_weights()

    def init_weights(self):
        weights = []
        for i in range(len(self.units) - 1):
            wi = torch.empty((self.units[i], self.units[i + 1]), device=device, dtype=dtype, requires_grad=True)
            torch.nn.init.normal_(wi, mean=0.0, std=self.weight_scale / np.sqrt(self.units[i]))
            weights.append(wi)
        return weights

    def init_recurrent_weights(self):
        recurrent_weights = []
        for i in range(1, len(self.units)):
            vi = torch.empty((self.units[i], self.units[i]), device=device, dtype=dtype, requires_grad=True)
            torch.nn.init.normal_(vi, mean=0.0, std=self.weight_scale / np.sqrt(self.units[i]))
            recurrent_weights.append(vi)
        return recurrent_weights

    def run_recurrent(self, inputs, batch_size, steps, spike_fn=SurrGradSpike.apply):
        layer_outputs = [inputs]
        mem_recs = []
        spike_recs = []

        for i in range(1, len(self.units) - 1):
            hidden_units = self.units[i]
            hi = torch.zeros((batch_size, hidden_units), device=device, dtype=dtype)
            hi_from_prev_layer = torch.einsum('abc, cd -> abd', (layer_outputs[i - 1], self.weights[i - 1]))
            syn_hidden_i = torch.zeros((batch_size, hidden_units), device=device, dtype=dtype)  # synapses
            mem_hidden_i = torch.zeros((batch_size, hidden_units), device=device, dtype=dtype)  # membranes

            mem_rec_hidden_i = [mem_hidden_i]
            spike_rec_hidden_i = [mem_hidden_i]

            for dt in range(steps):
                hi = hi_from_prev_layer[:, dt] + torch.einsum('ab, bc -> ac', (hi, self.recurrent_weights[i-1]))
                mem_threshold = mem_hidden_i - 1.0
                spike_out = spike_fn(mem_threshold)
                rst = torch.zeros_like(mem_hidden_i)
                c = (mem_threshold > 0)
                rst[c] = torch.ones_like(mem_hidden_i)[c]

                new_syn = self.alpha * syn_hidden_i + hi[:, dt]
                new_mem = self.beta * mem_hidden_i + syn_hidden_i - rst
                mem_hidden_i = new_mem
                syn_hidden_i = new_syn

                mem_rec_hidden_i.append(mem_hidden_i)
                spike_rec_hidden_i.append(spike_out)

            spike_rec_hidden_i = torch.stack(spike_rec_hidden_i, dim=1)
            mem_rec_hidden_i = torch.stack(mem_rec_hidden_i, dim=1)

            layer_outputs.append(spike_rec_hidden_i)  # append output so it can be fed to the next hidden layer
            mem_recs.append(mem_rec_hidden_i)  # append records
            spike_recs.append(spike_rec_hidden_i)  # append records

        # readout layer
        hn = torch.einsum('abc, cd -> abd', (layer_outputs[-1], self.weights[-1]))
        flt = torch.zeros((batch_size, self.units[-1]), device=device, dtype=dtype)
        spike_out = torch.zeros((batch_size, self.units[-1]), device=device, dtype=dtype)

        out_rec = [spike_out]
        for dt in range(steps):
            new_flt = self.alpha * flt + hn[:, dt]
            new_out = self.beta * out + flt

            flt = new_flt
            out = new_out

            out_rec.append(out)

        out_rec = torch.stack(out_rec, dim=1)
        return out_rec, spike_recs, layer_outputs, mem_recs


    def run_feed_forward(self, inputs, batch_size, steps, spike_fn=SurrGradSpike.apply):
        layer_outputs = [inputs]
        mem_recs = []
        spike_recs = []

        # compute activity of every hidden layer
        # hidden layers are stored from index 1 to index len(self.units) - 2
        for i in range(1, len(self.units) - 1):
            hidden_units = self.units[i]  # units in next layer

            # sum of output from previous layer and weights of current hidden layer
            hi = torch.einsum('abc, cd -> abd', (layer_outputs[i - 1], self.weights[i - 1]))
            syn_hidden_i = torch.zeros((batch_size, hidden_units), device=device, dtype=dtype)  # synapses
            mem_hidden_i = torch.zeros((batch_size, hidden_units), device=device, dtype=dtype)  # membranes

            mem_rec_hidden_i = [mem_hidden_i]
            spike_rec_hidden_i = [mem_hidden_i]

            for dt in range(steps):
                mem_threshold = mem_hidden_i - 1.0
                spike_out = spike_fn(mem_threshold)
                rst = torch.zeros_like(mem_hidden_i)
                c = (mem_threshold > 0)
                rst[c] = torch.ones_like(mem_hidden_i)[c]

                new_syn = self.alpha * syn_hidden_i + hi[:, dt]
                new_mem = self.beta * mem_hidden_i + syn_hidden_i - rst
                mem_hidden_i = new_mem
                syn_hidden_i = new_syn

                mem_rec_hidden_i.append(mem_hidden_i)
                spike_rec_hidden_i.append(spike_out)

            spike_rec_hidden_i = torch.stack(spike_rec_hidden_i, dim=1)
            mem_rec_hidden_i = torch.stack(mem_rec_hidden_i, dim=1)

            layer_outputs.append(spike_rec_hidden_i)  # append output so it can be fed to the next hidden layer
            mem_recs.append(mem_rec_hidden_i)  # append records
            spike_recs.append(spike_rec_hidden_i)  # append records

        # readout layer
        hn = torch.einsum('abc, cd -> abd', (layer_outputs[-1], self.weights[-1]))
        flt = torch.zeros((batch_size, self.units[-1]), device=device, dtype=dtype)
        spike_out = torch.zeros((batch_size, self.units[-1]), device=device, dtype=dtype)

        out_rec = [spike_out]
        for dt in range(steps):
            new_flt = self.alpha * flt + hn[:, dt]
            new_out = self.beta * spike_out + flt
            flt = new_flt
            spike_out = new_out
            out_rec.append(spike_out)

        out_rec = torch.stack(out_rec, dim=1)
        return out_rec, spike_recs, layer_outputs, mem_recs

    def train(self, x_data, y_data, batch_size, num_steps=100, time_step=1e-3, lr=1e-3, num_epochs=10,
              use_regularizer=False):

        if not use_regularizer:
            optimizer = torch.optim.Adam(self.weights, lr=lr, betas=(0.9, 0.999))
        else:
            optimizer = torch.optim.Adamax(self.weights, lr=lr, betas=(0.9, 0.999))
        log_softmax_fn = nn.LogSoftmax(dim=1)
        loss_fn = nn.NLLLoss()

        loss_hist = []
        for epoch in range(num_epochs):
            local_loss = []
            for x_local, y_local in sparse_data_generator(x_data, y_data, batch_size, num_steps, self.units[0],
                                                          time_step):
                # todo simplify
                if not self.recurrent_weights:
                    output, spike_recs, layer_outputs, mem_recs = self.run_feed_forward(x_local.to_dense(), batch_size,
                                                                                        num_steps)
                else:
                    output, spike_recs, layer_outputs, mem_recs = self.run_recurrent(x_local.to_dense(), batch_size,
                                                                                     num_steps)

                output_max, _ = torch.max(output, 1)
                log_p_y = log_softmax_fn(output_max)

                loss_val = loss_fn(log_p_y, y_local)
                if use_regularizer:
                    flattened_spike_recs = torch.stack(spike_recs, dim=0)
                    reg_loss = 1e-5 * torch.sum(
                        flattened_spike_recs)  # todo tezko rict, jestli secist vsechny layery, posledni nebo ktery
                    reg_loss += 1e-5 * torch.mean(torch.sum(
                        torch.sum(flattened_spike_recs, dim=0), dim=0
                    ) ** 2)
                    loss_val += reg_loss

                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()
                local_loss.append(loss_val.item())

            mean_loss = np.mean(local_loss)
            print("Epoch %i: loss=%.5f" % (epoch + 1, mean_loss))
            loss_hist.append(mean_loss)

        return loss_hist


def compute_classification_accuracy(x_data, y_data, batch_size, snn_model: DeepSNNModel, time_step=1e-3, num_steps=100):
    accs = []
    for x_local, y_local in sparse_data_generator(x_data, y_data, batch_size, num_steps, snn_model.units[0], time_step,
                                                  False):
        if not snn_model.recurrent_weights:
            output, spike_recs, layer_outputs, mem_recs = snn_model.run_feed_forward(x_local.to_dense(), batch_size,
                                                                                     num_steps)
        else:
            output, spike_recs, layer_outputs, mem_recs = snn_model.run_recurrent(x_local.to_dense(), batch_size,
                                                                                  num_steps)

        output_max, _ = torch.max(output, 1)
        _, output_argmax = torch.max(output_max, 1)
        tmp = np.mean((y_local == output_argmax).detach().cpu().numpy())
        accs.append(tmp)
    return np.mean(accs)


dataset_folder = os.path.join('cached_datasets')

train_dataset = torchvision.datasets.MNIST(dataset_folder, train=True,
                                           transform=None, target_transform=None, download=True)
test_dataset = torchvision.datasets.MNIST(dataset_folder, train=False,
                                          transform=None, target_transform=None, download=True)

# Standardize data
x_train = np.array(train_dataset.data, dtype=np.float)
x_train = x_train.reshape(x_train.shape[0], -1) / 255
x_test = np.array(test_dataset.data, dtype=np.float)
x_test = x_test.reshape(x_test.shape[0], -1) / 255

y_train = np.array(train_dataset.targets, dtype=np.int)
y_test = np.array(test_dataset.targets, dtype=np.int)

units = [28 * 28, 100, 150, 250, 10]
model = DeepSNNModel(units)

print('Feedforward DSNN: 100 units in 1st hidden layer, 150 units in 2nd hidden layer, 250 units in third hidden layer')
model.train(x_train, y_train, 256, num_epochs=15)
print('Classification accuracy on training data: {:.5f}'.format(
    compute_classification_accuracy(x_train, y_train, 256, model)))
print('Classification accuracy on testing data: {:.5f}'.format(
    compute_classification_accuracy(x_test, y_test, 256, model)))
