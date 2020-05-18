import torch
import numpy as np

TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class CartPoleConfig:

    # Create and move this tensor to GPU so that
    # we do not waste time moving it repeatedly to GPU later
    ee_sub = torch.tensor([0.0, 0.6], device=TORCH_DEVICE, dtype=torch.float)

    @staticmethod
    def preprocess_obs(obs):
        if isinstance(obs, np.ndarray):
            return np.concatenate([np.sin(obs[:, 1:2]), np.cos(obs[:, 1:2]), obs[:, :1], obs[:, 2:]], axis=1)
        elif isinstance(obs, torch.Tensor):
            return torch.cat([
                obs[:, 1:2].sin(),
                obs[:, 1:2].cos(),
                obs[:, :1],
                obs[:, 2:]
            ], dim=1)

    @staticmethod
    def postprocess_obs(obs):
        return obs

    # @staticmethod
    # def get_reward(obs_and_action):
    #     ee_pos = CartPoleConfig.get_end_effector_pos(obs_and_action)
    #     ee_pos -= CartPoleConfig.ee_sub # Substract bias
    #     ee_pos = ee_pos ** 2
    #     ee_pos = - ee_pos.sum(dim=1)
    #     return (ee_pos / (0.6 ** 2)).exp()

    @staticmethod
    def get_reward(obs, action):
        # return obs_and_action[:, 2]**3 - 0.8*obs_and_action[:, 0]**2
        return torch.cos(obs[:, 2]) - 0.01*(obs[:, 0]**2)

    @staticmethod
    def get_end_effector_pos(obs):
        x0, theta = obs[:, :1], obs[:, 1:2]

        return torch.cat([
            x0 - 0.6 * theta.sin(), -0.6 * theta.cos()
        ], dim=1)
