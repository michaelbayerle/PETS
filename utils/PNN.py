import numpy as np
import torch
import torch.nn as nn
import torch.utils
from torch.utils import data

import utils.nn_utils as nn_utils


class PNN(nn.Module):
    """Probabilistic neural network"""
    def __init__(self, ob_dim, n_acts, is_discrete, hidden_size, numLayers=1):
        super(PNN, self).__init__()
        self.ob_dim = ob_dim
        self.n_acts = n_acts
        self.is_discrete = is_discrete
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(ob_dim + n_acts, hidden_size))
        for _ in range(numLayers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, ob_dim * 2))

        # Apply random weights to all the layers
        self.apply(nn_utils.init_weights)
        self.statistics = dict() # Data statistics for normalizing input and output

    def forward(self, obs, ac):
        normalized_obs = nn_utils.normalize_obs(obs, self.statistics)
        
        # If the action is discrete, the action needs to be one hot encoded
        if self.is_discrete:
            ac = ac.long()
            one_hot = torch.zeros(ac.size(0), self.n_acts, device=nn_utils.DEVICE)
            ac_onehot = one_hot.scatter_(1, ac.data, 1)
            x = torch.cat((normalized_obs, ac_onehot), 1)
        else:
            x = torch.cat((normalized_obs, ac), 1)

        for i in range(len(self.layers) - 1):
            # x = F.relu(self.layers[i](x))
            # use the swish-1 / silu activation function like PETS paper
            x = nn_utils.silu(self.layers[i](x))

        nn_output = self.layers[len(self.layers) - 1](x)

        normalized_mean_deltas = nn_output[:, 0:self.ob_dim]
        normalized_logvar_deltas = nn_output[:, self.ob_dim:]
        normalized_var_deltas = torch.exp(normalized_logvar_deltas)
        
        mean_deltas = nn_utils.unnormalize_deltas(normalized_mean_deltas, self.statistics)

        return mean_deltas, normalized_mean_deltas, normalized_var_deltas

    def predict(self, obs, ac):
        """
        Used by MPCpolicy do_control to only return the predicted state change
        """
        mean_deltas, _, _ = self(obs, ac)
        return mean_deltas

    def gaussian_loss(self, mean, target, var):
        loss = torch.sum(var.reciprocal() * ((mean - target) ** 2), 1) + torch.prod(var, 1)
        return loss.mean()

    def train_net(self, optimizer, epochs, batch_size, data_train, data_test, log=True):
        """
        Train the model and validate on the test data
        """
        train_generator = data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
        test_generator = data.DataLoader(data_test, batch_size=batch_size, shuffle=True)
        train_losses = []
        test_losses = []

        for epoch in range(epochs):
            train_loss = []
            for X_batch, delta_targets in train_generator:
                # Actions need to be casted into long to have integer representation
                ob_batch = X_batch[:, :self.ob_dim]
                ac_batch = X_batch[:, self.ob_dim:]
                _, deltas_mean_normalized, deltas_var_normalized = \
                    self(ob_batch, ac_batch)

                # Compute the loss wrt the normalized deltas, the input features will be
                # normalized int the fordward pass of the ANN
                delta_targets_normalized = nn_utils.normalize_deltas(delta_targets, self.statistics)
                model_loss = self.gaussian_loss(deltas_mean_normalized,
                                                delta_targets_normalized,
                                                deltas_var_normalized)

                train_loss.append(model_loss.item())
                optimizer.zero_grad()
                model_loss.backward()
                optimizer.step()

            train_loss = np.mean(train_loss)
            train_losses.append(train_loss)

            # Compute train_loss
            test_loss = []
            for X_batch, delta_targets in test_generator:
                with torch.no_grad():
                    ob_batch = X_batch[:, :self.ob_dim]
                    ac_batch = X_batch[:, self.ob_dim:]
                    _, deltas_mean_normalized, deltas_var_normalized = \
                        self(ob_batch, ac_batch)

                    # Compute the loss wrt the normalized deltas, the input features will be
                    # normalized int the fordward pass of the ANN
                    delta_targets_normalized = nn_utils.normalize_deltas(delta_targets, self.statistics)
                    model_loss = self.gaussian_loss(deltas_mean_normalized,
                                                    delta_targets_normalized,
                                                    deltas_var_normalized)
                test_loss.append(model_loss.item())
            test_loss = np.mean(test_loss)
            test_losses.append(test_loss)

            if log:
                print('Epoch: %3d/%3d \t train_loss: %.5f \t test_loss: %.5f'
                      % (epoch + 1, epochs, train_loss, test_loss), flush=True)
        return train_losses, test_losses

    def set_statistics(self, statistics):
        """ Set data statistics for normalizing input and output
        """
        self.statistics = statistics
