import random
import torch
from torch.utils import data
import numpy as np
from tqdm import tqdm

class DataGenerator(object):
    def __init__(self, args, env, device, mpc_policy, random_policy, max_size=1000000, split=0.8):
        """
        @member env: openai gym env instance
        @member args: dict of arguments
        @member mpc_policy: agent class, takes an obs and returns an action
        @member random_policy: random agent class
        """
        self.env = env
        self.args = args
        self.device = device
        self.mpc_policy = mpc_policy
        self.random_policy = random_policy
        self.replay_buffer = []
        self.max_size = max_size
        self.split = split

    def generate_data(self, policy, dataset_size, is_render):
        """
        Generate data using the given agent function
        Returns a list of episodes, where each episode is
        a trajectory [(observation, action, diff_observation)]
        """
        episodes = []

        all_rewards = []
        all_lengths = []

        generated_samples = 0
        episode_id = 0
        with tqdm(total=dataset_size, desc='Generating data') as pbar:
            while generated_samples < dataset_size:
                observation = self.env.reset().astype(np.float32)
                episode = []
                rewards = []

                done = False
                episode_steps = 0
                while (not done) and episode_steps < self.args.ep_len:
                    # Render first episode only
                    if is_render:
                        self.env.render()

                    action = policy.do_control(observation)
                    new_observation, reward, done, _ = self.env.step(action)
                    new_observation = new_observation.astype(np.float32)

                    episode.append((observation, action, new_observation, reward))
                    rewards.append(reward)
                    observation = new_observation
                    generated_samples += 1
                    episode_steps += 1
                    pbar.update()

                episodes.append(episode)
                all_rewards.append(np.sum(rewards))
                all_lengths.append(len(rewards))
                episode_id += 1

        return episodes, all_rewards, all_lengths

    def generate_random_data(self):
        print("Generating random data...")
        episodes, all_rewards, all_lengths = self.generate_data(
            self.random_policy, self.args.dataset_rand_size, False)
        self.replay_buffer.extend([item for episode in episodes for item in episode])
        self.replay_buffer = self.replay_buffer[-self.max_size:]
        self.update_statistics()
        return episodes, all_rewards, all_lengths

    def generate_closed_loop_data(self, render=False):
        print("Generating closed loop data...")
        episodes, all_rewards, all_lengths = self.generate_data(
            self.mpc_policy, self.args.dataset_rl_size, render)
        self.replay_buffer.extend([item for episode in episodes for item in episode])
        self.replay_buffer = self.replay_buffer[-self.max_size:]
        self.update_statistics()
        return episodes, all_rewards, all_lengths

    def generate_evaluation_data(self, is_random_policy=False, render=False):
        print("Generating evaluation data...")
        # Do not add to replay buffer
        if is_random_policy:
            return self.generate_data(
                self.random_policy, self.args.eval_batch_size, self.args.render)
        else:
            return self.generate_data(
                self.mpc_policy, self.args.eval_batch_size, render)

    def update_statistics(self):
        obs = [obs for obs, _, _, _ in self.replay_buffer]
        deltas = [nobs - obs for obs, _, nobs, _ in self.replay_buffer]

        obs_torch = torch.FloatTensor(obs).to(self.device)
        deltas_torch = torch.FloatTensor(deltas).to(self.device)

        self.statistics = {
            'ob_mean' : obs_torch.mean(dim=0),
            'ob_std' : obs_torch.std(dim=0),
            'delta_mean' : deltas_torch.mean(dim=0),
            'delta_std' : deltas_torch.std(dim=0)
        }

    def get_all_episodes(self):
        return self.all_episodes

    def get_dataset(self):
        return self.make_datasets()

    def make_datasets(self, dataset_size=None, random_sampling=False):
        """
        Generate torch dataset from the given episodes
        """
        if dataset_size is None:
            dataset_size = len(self.replay_buffer)

        if random_sampling:
            samples = random.choices(self.replay_buffer, k=dataset_size)
        else:
            samples = self.replay_buffer[-dataset_size:]

        X = np.array([np.append(observation, action)
                      for observation, action, _, _ in samples])
        y = np.array([new_observation - observation
                      for observation, _, new_observation, _ in samples])

        X = torch.from_numpy(X).float().to(self.device)
        y = torch.from_numpy(y).float().to(self.device)

        split_id = int(dataset_size * self.split)
        data_train = data.TensorDataset(X[:split_id, :], y[:split_id, :])
        data_test = data.TensorDataset(X[split_id:, :], y[split_id:, :])

        return data_train, data_test