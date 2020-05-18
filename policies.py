import gym
import numpy as np
import torch


class RandomPolicy(object):
    def __init__(self, env):
        self.env = env
        self.is_discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.ac_dim = env.action_space.n if self.is_discrete else env.action_space.shape[0]

    def do_control(self, obs):
        return self.env.action_space.sample()

class MPCPolicy(object):
    def __init__(self, args, env, ensemble, reward_function, device):
        self.args = args
        self.env = env
        self.ensemble = ensemble
        self.reward_function = reward_function
        self.device = device
        self.mpc_samples = args.mpc_samples
        self.mpc_horizon = args.mpc_horizon
        self.ob_dim = env.observation_space.shape[0]
        self.is_discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.ac_dim = env.action_space.n if self.is_discrete else env.action_space.shape[0]
        self.epsilon = 0.0

        if not self.is_discrete:
            self.low = env.action_space.low
            self.high = env.action_space.high

        self.args = args
        self.gamma = args.gamma # discount factor

    def update_epsilon(self, iteration):
        self.epsilon = self.args.epsilon_decay ** (iteration + 1)
        return self.epsilon

    def do_control(self, observation):
        """
        Given the current observation, return the best action:
        Generates policy_samples random trajectories of length policy_look_ahead,
        performs rollout using the current model, and return the first action which
        leads to the best state, according to the reward function
        """
        if np.random.rand(1) <= self.epsilon:
            return self.env.action_space.sample()

        # TODO: incorporate low and high into the sampling
        # TODO: move this functional
        if "CartPole" in self.args.env_name:
            actions = torch.FloatTensor(
               size=(self.args.mpc_samples,
                     self.args.mpc_horizon, 1)).random_(0, self.ac_dim).to(self.device)
        elif "Acrobot" in self.args.env_name:
            actions = torch.FloatTensor(
               size=(self.args.mpc_samples,
                     self.args.mpc_horizon, 1)).random_(0, self.ac_dim).to(self.device)
        elif "Pendulum" in self.args.env_name:
            actions = torch.FloatTensor(
               size=(self.args.mpc_samples,
                     self.args.mpc_horizon,
                     self.ac_dim)).uniform_(-2.0, 2.0).to(self.device)
        elif "MountainCar" in self.args.env_name:
            actions = torch.FloatTensor(
               size=(self.args.mpc_samples,
                     self.args.mpc_horizon,1)).random_(0, self.ac_dim).to(self.device)
        else:
            actions = torch.FloatTensor(
               size=(self.args.mpc_samples,
                     self.args.mpc_horizon,
                     self.ac_dim)).uniform_(-1.0, 1.0).to(self.device)

        last_state = torch.zeros((self.mpc_samples, self.ob_dim), device=self.device)

        # Assign the same observation to all the rows
        last_state[:, :] = torch.tensor(observation, device=self.device)
        rewards = torch.zeros(self.args.mpc_samples, device=self.device)

        # rollout
        for t in range(self.args.mpc_horizon):

            with torch.no_grad():
                # predict next states
                pred = self.ensemble.predict(last_state, actions[:, t, :])
                last_state += pred
                rewards += (self.args.gamma**t) * self.reward_function(last_state, actions[:, t, :])

        best_act_seq = actions[torch.argmax(rewards)]
        best_first_act = best_act_seq[0]

        if not self.is_discrete:
            return best_first_act.cpu()
        else:
            return int(best_first_act[0])
