import glob
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

from utils.DNN import DNN
from utils.PNN import PNN


class Ensemble:
    def __init__(self, ob_dim, n_actions, is_discrete, pnn=False, ensemble_size=3, lr=0.001, hidden_size=50,
                 hidden_layers=2, device=None):
        super(Ensemble, self).__init__()
        self.size = ensemble_size
        self.device = device
        self.is_discrete = is_discrete
        self.n_acts = n_actions

        print("PNN: ", pnn)
        if pnn:
            self.models = [PNN(ob_dim, n_actions, is_discrete, hidden_size, hidden_layers).to(device)
                           for i in range(ensemble_size)]
        else:
            self.models = [DNN(ob_dim, n_actions, is_discrete, hidden_size, hidden_layers).to(device)
                           for i in range(ensemble_size)]
        self.optimizers = [optim.Adam(model.parameters(), lr=lr) for model in self.models]
        self.statistics = dict() # Data statistics for normalizing input and output

    def predict(self, obs, ac):
        predictions = []

        # Iterate over the neural networks and get the predictions
        for model in self.models:
            deltas = model.predict(obs, ac)
            predictions.append(deltas)

        predictions_tensor = torch.stack(predictions)
        mean_prediction = torch.mean(predictions_tensor, 0)
        return mean_prediction

    def __call__(self, obs, ac):
        deltas, _ = self.predict(obs, ac)
        return deltas

    def train_net(self, epochs, batch_size, data_generator, samples_per_model=None):
        # Update the statistics
        self.set_statistics(data_generator.statistics)

        # Train all the networks in the ensemble
        train_losses = []
        test_losses = []
        for i, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
            print("TRAINING MODEL %2d/%2d" % (i + 1, self.size))
            data_train, data_test = data_generator.make_datasets(dataset_size=samples_per_model, random_sampling=True)
            train_loss, test_loss = model.train_net(optimizer, epochs, batch_size, data_train, data_test)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
        return train_losses, test_losses

    def load_weights(self, weights_dir):
        """
        Load weights
        Usage: weights_dir = '../data/mb_Acrobot/models/rep_1_*'
        """

        weights_paths = glob.glob(weights_dir)
        print('Loading the following weights:')
        print(weights_paths)
        assert len(weights_paths) == len(self.models)

        for (model, weights_path) in zip(self.models, weights_paths):
            model.load_state_dict(torch.load(weights_path, map_location=self.device))
            model.eval()

    def set_statistics(self, statistics):
        """ Set up statistics for normalizing input and output of neural network
        """
        self.statistics = statistics
        # Update the statistics over all the networks in the ensemble
        for i, model in enumerate(self.models):
            model.set_statistics(statistics)
