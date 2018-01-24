import tensorflow as tf
import numpy as np
from gym import wrappers

import argparse
import sys
import json

from categorical_algorithm import CategoricalAlgorithm
from environment import AtariEnvironment


class EvalRunner(object):
    def __init__(self, model, env, epsilon=0.01, verbose=False):
        self.model = model
        self.env = env
        self.eps = epsilon
        self.moving_average = 0
        self.num_episodes = 0
        self.verbose = verbose

    def run_episode(self, max_frames=None):
        s, R, is_terminal, _ = self.env.random_start()

        states = [s] * state_history

        X0 = np.concatenate(states, axis=2)

        total_reward = 0
        total_points = 0

        i = 0

        while not is_terminal and i and i < max_frames:
            q_vals = model.q_values([X0], sess)[0]

            if np.random.rand() > self.epsilon:
                action = q_vals.argmax()
            else:
                action = np.random.randint(env.action_space.n)

            s, R, is_terminal, _ = env.step(action)

            states.append(s)

            X0 = np.concatenate(states[1:], axis=2)

            total_reward += R
            total_points += np.sign(R)
            i += 1

            states = states[1:]

            if self.verbose:
                print('%s - %s - %s - %s\r' % (i, total_reward, action, q_vals),)

        if self.verbose:
            print()
            print('Greedy Validation Score: %s - Total Reward Events: %s' % (total_reward, total_points))

        self.num_episodes += 1
        self.moving_average += 0.99 * self.moving_average + 0.01 * total_reward

        return total_reward, total_points

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Which config to load the model from')
    parser.add_argument('--game', default='breakout', type=str, help='The Atari game to run')
    parser.add_argument('--checkpoint-dir', type=str, default='/tmp/categorical_model',
                        help='The directory where the model will be stored.')
    parser.add_argument('--eps', type=float, help='What is the random action rate during evaluation', default=0.01)
    parser.add_argument('--render', type=bool, default=False, help='Should we render output?')

    args = parser.parse_args(sys.argv[1:])

    with open(args.config) as infile:
        config = json.load(infile)

    input_size = config['input_size']
    state_history = config['state_history']
    n_atoms = config['n_atoms']
    v_min, v_max = config['v_bounds']
    model_checkpoints_dir = args.checkpoint_dir
    frame_skip = config.get('frame_skip', 4)
    fire_start = config.get('fire_start', False)

    env = AtariEnvironment(args.game, frame_skip, fire_start=fire_start, render=args.render)

    n_actions = env.action_space.n

    print 'Creating Model Graphs...'
    cat = CategoricalAlgorithm(config, n_actions=env.action_space.n, name='cat')

    init = tf.global_variables_initializer()
    sess = tf.Session()

    print 'Initialising Variables...'
    sess.run(init)

    saver = tf.train.Saver()

    latest = tf.train.latest_checkpoint(model_checkpoints_dir)
    if latest is not None:
        print('Restoring model from: %s' % latest)
        saver.restore(sess, latest)
    else:
        print('Unable to find latest checkpoint. Starting fresh.')

    runner = EvalRunner(cat, env, epsilon=args.eps, verbose=True)
    try:
        while True:
            runner.run_episode()
    except KeyboardInterrupt:
        print('Done evaluation')
        print('Total Episodes: %s - 100 episode moving average: %s' % (runner.num_episodes, runner.moving_average))
