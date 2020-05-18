import argparse
import os
import pdb
import time
from collections import OrderedDict

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch

import sunblaze_envs
import utils.nn_utils as nn_utils
from data_generator import DataGenerator
from env_config.CartPoleConfig import CartPoleConfig
from env_config.PendulumConfig import PendulumConfig
from logger import Logger
from policies import MPCPolicy, RandomPolicy
from utils.ensemble import Ensemble
from utils.PNN import PNN


def MPE(env, current_episodes, model, horizon, plot=True, label=None, max_frames=500):
    """
    Generate Mean Prediction Error
    For each observation, predict the next observation using the given actions
    """
    ac_dim = model.n_acts

    mpe_losses = [[] for _ in range(horizon)]

    frames = 0
    for episode in current_episodes:

        # Compute MPE starting from each state
        for start in range(len(episode)):

            observation, _, _, _ = episode[start]
            # the first prediction is the true observation (add batch dimension)
            prediction = torch.tensor([observation], device=nn_utils.DEVICE)
            action = torch.zeros((1, ac_dim), device=nn_utils.DEVICE)

            # predict the next states
            for h in range(min(horizon, len(episode) - start - 1)):
                observation, a, new_observation, _ = episode[start + h]
                action[:] = torch.tensor(a, device=nn_utils.DEVICE)
                expected_observation = torch.tensor(new_observation, device=nn_utils.DEVICE)

                with torch.no_grad():
                    if isinstance(model, PNN):
                        pred_diff, var = model(prediction, action)
                    else:
                        pred_diff = model(prediction, action)
                    prediction += pred_diff

                mpe_losses[h].append(torch.norm(expected_observation - prediction).numpy())

            frames += 1
            if frames > max_frames:
                break
        if frames > max_frames:
            break

    if plot:
        # Compute the mean for each horizon
        mpe_means = [np.mean(loss) for loss in mpe_losses]
        fig = plt.figure(2)
        plt.xlabel('Horizon')
        plt.ylabel('MSE')
        plt.title('Model Prediction Error')
        plt.plot(mpe_means, label=label)
        plt.legend()
        plt.show()
        plt.pause(0.1)

    return mpe_losses


def get_success_function(env):
    name = env.unwrapped.spec.id

    if 'CartPole' in name:
        return lambda final_time_step: final_time_step > 195
    elif 'Acrobot' in name:
        return lambda final_time_step: final_time_step <= 81
    elif 'MountainCar' in name:
        return lambda final_time_step: final_time_step < 110
    elif 'Pendulum' in name:
        # not actually used
        return lambda final_time_step: final_time_step < 200


def get_reward_function(env):
    name = env.unwrapped.spec.id

    if 'CartPole' in name:
        return CartPoleConfig.get_reward
    elif 'Pendulum' in name:
        return PendulumConfig.get_reward
    elif 'Acrobot' in name:
        return lambda state, _: -state[:, 0] - (state[:, 0]*state[:, 2] - state[:, 1]*state[:, 3])
    elif 'MountainCar' in name:
        # Go as much to the right as possible (x)
        return lambda state, _: state[:, 0]
    elif name == 'InvertedPendulumPyBulletEnv-v0':
        # cos(theta)^3 - 0.01 * x^2
        return lambda state, _: state[:, 2] ** 3 - 0.8 * state[:, 0] ** 2
    elif name == 'InvertedDoublePendulumPyBulletEnv-v0':
        # TODO: find better reward function
        # cos(theta) * cos(gamma)
        return lambda state, _: state[:, 3] * state[:, 6]
    elif name == 'InvertedPendulumSwingupPyBulletEnv-v0':
        # cos(theta)^2 * sign(cos(theta))
        return lambda state, _: state[:, 2] ** 3 - 0.8 * state[:, 0] ** 2
    else:
        raise NotImplementedError('Reward function is not implemented for the given environment')


def main(args, logdir):
    """
    Model Based Reinforcement Learning
    1) Generate random trajectories
    2) Train the model on the generated data
    3) For each repetition:
        a) Generate new data using the MPC controller
        b) Retrain the model using the new data and the old data
        c) (Optional) Compute Mean Prediction Error
    """

    # SETUP
    train_envs = []
    test_envs = []
    if args.no_sunblaze:
        train_env = gym.make(args.env_name)
        test_env = gym.make(args.env_name)

        if 'PyBullet' in args.env_name and args.render:
            train_env.render()
            train_env.reset()

    elif args.test_type == 'interpolation':
        train_envs.append(sunblaze_envs.make('Sunblaze' + args.env_name + 'RandomNormal-v0'))
        test_envs.append(sunblaze_envs.make('Sunblaze' + args.env_name + 'RandomNormal-v0'))

    elif args.test_type == 'extrapolation':
        train_envs.append(sunblaze_envs.make('Sunblaze' + args.env_name + '-v0'))
        train_envs.append(sunblaze_envs.make('Sunblaze' + args.env_name + 'RandomNormal-v0'))

        test_envs.append(sunblaze_envs.make('Sunblaze' + args.env_name + 'RandomExtreme-v0'))
        test_envs.append(sunblaze_envs.make('Sunblaze' + args.env_name + 'RandomNormal-v0'))

    else:
        train_envs.append(sunblaze_envs.make('Sunblaze' + args.env_name + '-v0'))
        test_envs.append(sunblaze_envs.make('Sunblaze' + args.env_name + '-v0'))

    test_cnt = 0
    for train_env in train_envs:

        assert isinstance(train_env.observation_space, gym.spaces.Box)

        start_time = time.time()
        logger = Logger(logdir)

        is_discrete = isinstance(train_env.action_space, gym.spaces.Discrete)

        ob_dim = train_env.observation_space.shape[0]
        ac_dim = train_env.action_space.n if is_discrete else train_env.action_space.shape[0]

        reward_function = get_reward_function(train_env)

        train_env.reset()
        ensemble = Ensemble(ob_dim, ac_dim, is_discrete, args.pnn, args.ensemble_size, args.lr, args.hidden_size, device=nn_utils.DEVICE)

        # TRAIN
        # Instantiate policies
        mpc_policy = MPCPolicy(args, train_env, ensemble, reward_function, nn_utils.DEVICE)
        random_policy = RandomPolicy(train_env)

        # Instantiate Data generator
        data_generator = DataGenerator(args, train_env, nn_utils.DEVICE, mpc_policy, random_policy, max_size=args.max_memory_size)

        if args.weights_paths is not None:
            # If weights are given, visualize and quit
            ensemble.load_weights(args.weights_paths)

            current_episodes, rewards, lengths = data_generator.generate_closed_loop_data(args.render)
            if args.mpe:
                MPE(train_env, current_episodes, ensemble, args.mpc_horizon, label='Ensemble %s' % (args.weights_paths))
            print('avg reward episode %f' %(np.mean(rewards)))
            print('avg len %f' %(np.mean([len(ep) for ep in current_episodes])))
            return

        # Otherwise train model on random trajectories
        current_episodes, train_rewards, train_lengths = data_generator.generate_random_data()

        # Train initial model using random trajectories
        train_loss, test_loss = ensemble.train_net(args.epochs_rand, args.batch_size, data_generator, samples_per_model=args.samples_per_model)

        if args.mpe:
            print('Computing MPE')
            for (i, model) in enumerate(ensemble.models):
                MPE(train_env, current_episodes, model, args.mpc_horizon, label='random data, model %d' %(i))
            if len(ensemble.models) > 1:
                MPE(train_env, current_episodes, ensemble, args.mpc_horizon, label='random data, ensemble')


        _, eval_rewards, eval_lengths = data_generator.generate_evaluation_data(render=args.render)

        # TODO: keep test data only for test data
        for itr in range(args.repetitions):
            print('\nMPC Repetition %d / %d \n' % (itr + 1, args.repetitions))
            epsilon = mpc_policy.update_epsilon(itr)
            perform_logging(itr, logger,
                            eval_rewards, train_rewards, test_loss, train_loss,
                            eval_lengths, train_lengths, start_time, epsilon)
            current_episodes, train_rewards, train_lengths = data_generator.generate_closed_loop_data()

            train_loss, test_loss = ensemble.train_net(args.epochs_rl, args.batch_size, data_generator, samples_per_model=args.samples_per_model)

            if args.mpe:
                print('Computing MPE')
                for (i, model) in enumerate(ensemble.models):
                    MPE(train_env, current_episodes, model, args.mpc_horizon, label='rep %d, model %d' % (itr, i))
                if len(ensemble.models) > 1:
                    MPE(train_env, current_episodes, ensemble, args.mpc_horizon, label='rep %d, ensemble' %(itr))

            _, eval_rewards, eval_lengths = data_generator.generate_evaluation_data(render=args.render)

            if args.save_model:
                for (i, model) in enumerate(ensemble.models):
                    save_file = '%s/models/rep_%d_model_%d_%.4f.pt' % (str(logdir), itr, i, test_loss[i][-1])
                    torch.save(model.state_dict(), save_file)

        # SUNBLAZE TEST
        for test_env in test_envs:
            test_name = test_env.unwrapped.spec.id
            train_name = train_env.unwrapped.spec.id
            if test_cnt < 3:
                print('\nTESTING: ' + train_name + ' on ' + test_name, flush=True)
                success_function = get_success_function(test_env)
                num_success = 0
                rewards = []
                for ep_num in range(args.test_episodes):
                    success, ep_reward = run_test_episode(test_env, mpc_policy, success_function, args.render)
                    rewards.append(ep_reward)
                    num_success += int(success)
                    print('Test episode: %2d / %2d \t Success: %d \t Reward: %d' %
                          (ep_num + 1, args.test_episodes, int(success), ep_reward), flush=True)

                score = num_success / args.test_episodes * 100
                logger.log_scalar(score, test_name + '-' + train_name, 0)
                with open(train_name + '_' + test_name + '_score.txt', 'w+') as f:
                    f.write('Score for ' + train_name + ' tested on ' + test_name + ': ' + str(score))

                print('\nScore for ' + train_name + ' tested on ' + test_name + ' testing: ', score, flush=True)
                test_cnt += 1


def run_test_episode(env, policy, success_function, render):
    observation = env.reset()
    done = False
    rewards = []
    obs_list = [observation]
    episode_steps = 0
    while not done:
        if render:
            env.render()
        action = policy.do_control(observation)
        new_observation, reward, done, _ = env.step(action)
        new_observation = new_observation.astype(np.float32)
        rewards.append(reward)
        observation = new_observation
        obs_list.append(observation)
        episode_steps += 1

    if 'Pendulum' in env.unwrapped.spec.id:
        thetas = np.arccos(np.asarray(obs_list)[:, 0])
        success = np.all(np.abs(thetas[100:]) < np.pi/3)
    else:
        success = success_function(episode_steps)
    return success, np.sum(rewards)


def perform_logging(itr, logger, eval_returns, train_returns, eval_loss, train_loss,
                    eval_ep_lens, train_ep_lens, start_time, epsilon):
    """
    Function adapted from cs285 berkeley deep reinforcement learning
    """

    # decide what to log
    logs = OrderedDict()
    logs["Eval_AverageReturn"] = np.mean(eval_returns)
    logs["Eval_StdReturn"] = np.std(eval_returns)
    logs["Eval_MaxReturn"] = np.max(eval_returns)
    logs["Eval_MinReturn"] = np.min(eval_returns)
    logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)
    # logs["Eval_ModelLoss"] = eval_loss[-1]

    logs["Train_AverageReturn"] = np.mean(train_returns)
    logs["Train_StdReturn"] = np.std(train_returns)
    logs["Train_MaxReturn"] = np.max(train_returns)
    logs["Train_MinReturn"] = np.min(train_returns)
    logs["Train_AverageEpLen"] = np.mean(train_ep_lens)
    # logs["Train_ModelLoss"] = train_loss[-1]

    # logs["Train_EnvstepsSoFar"] = self.total_envsteps
    logs["TimeSinceStart"] = time.time() - start_time

    logs["epsilon"] = epsilon

    # perform the logging
    for key, value in logs.items():
        print('{} : {}'.format(key, value))
        logger.log_scalar(value, key, itr)
    print('Done logging...\n\n')

    logger.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, help='Name of experiment')
    parser.add_argument('--env_name', '--env', type=str, default='CartPole')
    parser.add_argument('--render', action='store_true', help='Render first episode')
    parser.add_argument('--mpe', action='store_true', help='Compute Mean Prediction Error')
    parser.add_argument('--pnn', action='store_true', help='Use a PNN')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--eval_batch_size', type=int, default=3000, help='Eval Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon_decay', type=float, default=0.0, help='Pick random action with probability (epsilon_decay^nr. iteration). Set to 0.0 for no randomness')
    parser.add_argument('--epochs_rand', type=int, default=5, help='Epochs to train the model on random actions')
    parser.add_argument('--epochs_rl', type=int, default=100, help='Epochs to train the model at each repetition')
    parser.add_argument('--dataset_rand_size', type=int, default=200,
                        help='Nr. of samples to train the model on random actions')
    parser.add_argument('--dataset_rl_size', type=int, default=1000,
                        help='Nr. of samples to train the model using mpc')
    parser.add_argument('--max_memory_size', type=int, default=80000,
                        help='Nr. of samples to store in the buffer memory')
    parser.add_argument('--samples_per_model', type=int, default=40000, help='Nr. of samples to train each model')
    parser.add_argument('--ep_len', type=int, default=500, help='Max Length of episode')
    parser.add_argument('--repetitions', type=int, default=20, help='A repetition consists of generating data using mpc and training')
    parser.add_argument('--hidden_size', type=int, default=500, help='Size of the hidden layer in the neural network')
    parser.add_argument('--hidden_layers', type=int, default=3, help='Number of hidden layers (excluded)')
    parser.add_argument('--ensemble_size', type=int, default=3, help='Number of networks')
    parser.add_argument('--mpc_samples', type=int, default=256, help='Nr. of trajectories')
    parser.add_argument('--mpc_horizon', type=int, default=10, help='Horizon')
    parser.add_argument('--save_model', action='store_true', help='Save the model weights')
    parser.add_argument('--weights_paths', type=str, help='Load the given weights model. Eg. \'../data/mb_Acrobot-v1_09-01-2020_14-28-24/models/rep_1_*\'')
    parser.add_argument('--test_episodes', type=int, default=1000, help='Nr. of episodes to test for sunblaze scoring')
    parser.add_argument('--test_type', type=str, default='default', help='Type of sunblaze test to run')
    parser.add_argument('--no_sunblaze', action='store_true', help='Do not use sunblaze general environments but exact name')

    args = parser.parse_args()

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    logdir_prefix = 'mb_'

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    exp_name = "" if args.exp_name is None else args.exp_name
    logdir = logdir_prefix + exp_name + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
        os.makedirs(logdir + '/models/')

    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    print(args)
    main(args, logdir)
