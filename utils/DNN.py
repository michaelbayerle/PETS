import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

import utils.nn_utils as nn_utils


class DNN(nn.Module):
    """Simple neural network"""
    def __init__(self, ob_dim, n_acts, is_discrete, hidden_size, numLayers=1):
        super(DNN, self).__init__()
        self.ob_dim = ob_dim
        self.n_acts = n_acts
        self.is_discrete = is_discrete
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(ob_dim + n_acts, hidden_size))
        for _ in range(numLayers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, ob_dim))

        self.apply(nn_utils.init_weights)
        self.statistics = dict() # Data statistics for normalizing input and output

    def forward(self, obs, ac):
        # The input to the network needs to be normalized
        normalized_obs = nn_utils.normalize_obs(obs, self.statistics)

        if self.is_discrete:
            # make one hot encoding
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
        normalized_deltas = self.layers[len(self.layers) - 1](x)
        
        # The MPC needs the unnormalized deltas to compute the predictions 
        deltas = nn_utils.unnormalize_deltas(normalized_deltas, self.statistics)
        return deltas, normalized_deltas 

    def predict(self, obs, action):
        """
        Used by MPCpolicy do_control to only return the predicted state change
        """
        deltas, _ = self(obs, action)
        return deltas 

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
                _, delta_normalized = self(ob_batch, ac_batch)

                # Compute the loss wrt the normalized deltas, the input features will be
                # normalized int the fordward pass of the ANN
                delta_targets_normalized = nn_utils.normalize_deltas(delta_targets, self.statistics)
                model_loss = F.mse_loss(delta_normalized, delta_targets_normalized)

                # Keep track of the train losses
                train_loss.append(model_loss.item())

                optimizer.zero_grad()
                model_loss.backward()
                optimizer.step()

            train_loss = np.mean(train_loss)
            train_losses.append(train_loss)

            # Compute test loss 
            test_loss = []
            for X_batch, delta_targets in test_generator:
                with torch.no_grad():
                    ob_batch = X_batch[:, :self.ob_dim]
                    ac_batch = X_batch[:, self.ob_dim:]
                    _, delta_normalized = self(ob_batch, ac_batch)
                    
                    # The model loss gets computed wrt the normalized deltas
                    delta_targets_normalized = nn_utils.normalize_deltas(delta_targets, self.statistics)
                    model_loss = F.mse_loss(delta_normalized, delta_targets_normalized)
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
