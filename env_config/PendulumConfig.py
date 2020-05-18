import torch
import numpy as np

TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class PendulumConfig:

    @staticmethod
    def preprocess_obs(obs):
        return obs

    @staticmethod
    def postprocess_obs(obs):
        return obs

    @staticmethod
    def get_reward(obs, action):
        th = torch.atan2(obs[:, 1], obs[:, 0])
        thdot = obs[:, 2]
        u = action.squeeze()
        costs = PendulumConfig.angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)
        return -costs

    @staticmethod
    def angle_normalize(x):
        return ((x+np.pi) % (2*np.pi)) - np.pi
