import tensorflow as tf
import numpy as np

import argparse
import sys
import json

from replay_buffer import ReplayBuffer
from categorical_algorithm import CategoricalAlgorithm, categorical_loss_target
from environment import AtariEnvironment
from evaluate import EvalRunner

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Which config to load the model from')
    parser.add_argument('--game', default='breakout', type=str, help='The Atari game to run')
    parser.add_argument('--checkpoint-dir', type=str, default='/tmp/categorical_model',
                        help='The directory where the model will be stored.')
    parser.add_argument('--model-name', type=str, default='categorical51', help='The name of the model to save as')

    args = parser.parse_args(sys.argv[1:])

    with open(args.config) as infile:
        config = json.load(infile)

    buffer_size = config['buffer_size']
    min_buffer_size = config['min_buffer_size']
    input_size = config['input_size']
    state_history = config['state_history']
    target_update_rate = config['target_update_rate']
    tau = (target_update_rate - 1.0) / target_update_rate
    target_update_mode = config.get('target_update_mode', 'every_n')
    n_atoms = config['n_atoms']
    v_min, v_max = config['v_bounds']
    max_eps = config['eps_start']
    min_eps = config['eps_min']
    eps_steps = config['eps_steps']
    eval_interval = config['evaluation_interval']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    model_checkpoints_dir = args.checkpoint_dir
    checkpoint_interval = config['checkpoint_interval']
    gamma = config['gamma']
    update_rate = config['update_rate']
    steps = config['steps']
    double_dqn = config.get('double_dqn', False)
    frame_skip = config.get('frame_skip', 4)
    fire_start = config.get('fire_start', False)
    clip_rewards = config.get('clip_rewards', False)

    history = ReplayBuffer(buffer_size=buffer_size, observation_window=state_history,
                           input_size=input_size)

    env = AtariEnvironment(args.game, frame_skip, fire_start=fire_start)

    print 'Creating Model Graphs...'
    cat = CategoricalAlgorithm(config, n_actions=env.action_space.n, name='cat')
    target = CategoricalAlgorithm(config, n_actions=env.action_space.n, name='target')

    model_vars = sorted([v for v in tf.global_variables() if v.name.startswith('cat')], key=lambda x: x.name)
    target_vars = sorted([v for v in tf.global_variables() if v.name.startswith('target')], key=lambda x: x.name)

    assign_ops = []
    for v, vt in zip(model_vars, target_vars):
        if target_update_mode == 'ewma':
            assign_ops.append(vt.assign(tau * vt + (1 - tau) * v))
        else:
            assign_ops.append(vt.assign(v))

    global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int32)
    inc = tf.assign_add(global_step, 1, name='increment')

    opt = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=0.0003125, beta1=0.95, beta2=0.95)

    train_step = tf.contrib.layers.optimize_loss(cat.loss, learning_rate=None, global_step=global_step,
                                                 optimizer=opt, clip_gradients=1.)

    astar_op = tf.gather_nd(target.probs, target.actions)
    init = tf.global_variables_initializer()
    sess = tf.Session()

    print 'Initialising Variables...'
    sess.run(init)

    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=8.0)

    latest = tf.train.latest_checkpoint(model_checkpoints_dir)
    if latest is not None:
        print('Restoring model from: %s' % latest)
        saver.restore(sess, latest)
    else:
        print('Unable to find latest checkpoint. Starting fresh.')

    states = []

    episodes = 0
    eval_score = 0

    graph = tf.get_default_graph()
    graph.finalize()

    frame_counter = 0

    eval_runner = EvalRunner(cat, env, epsilon=0.001, verbose=True)
    while True:
        step = sess.run(global_step)

        if step > steps:
            break

        s, R, is_terminal, info = env.random_start()
        is_done = is_terminal
        lives = info.get('ale.lives', 0)

        states = [s] * state_history

        X1 = np.concatenate(states, axis=2)

        print 'Starting episode %s' % episodes
        total_reward = 0

        while not is_terminal:
            frame_counter += 1
            X0 = X1

            eps = max(max_eps - float(step) * (max_eps - min_eps) / eps_steps, min_eps)

            if np.random.random() > eps:
                action = cat.get_actions([X0], sess)[0]
            else:
                action = np.random.randint(0, env.action_space.n)

            s, R, is_terminal, info = env.step(action)

            total_reward += R

            if clip_rewards:
                R = np.sign(R)

            if 0 < info.get('ale.lives', 0) < lives:
                is_done = True
                lives = info.get('ale.lives', 0)
            else:
                is_done = False

            states.append(s)
            X1 = np.concatenate(states[1:], axis=2)

            states = states[1:]

            history.add(s, action, R, is_terminal or is_done)

            if history.total_count > min_buffer_size and frame_counter % update_rate == 0:
                step = sess.run(global_step)

                if step % checkpoint_interval == 0:
                    saver.save(sess, os.path.join(model_checkpoints_dir, args.model_name), global_step=step)

                S0, actions, rewards, S1, terminal = history.sample_batch(batch_size)

                actions = list(enumerate(actions))
                rewards = np.array(rewards).reshape(batch_size, 1)

                if double_dqn:
                    q_vals = cat.q_values(S1, sess)
                else:
                    q_vals = target.q_values(S1, sess)

                a_star = np.array(list(enumerate(q_vals.argmax(1))))
                p_a_star = sess.run(astar_op, feed_dict={target.actions: a_star, target.input_layer: S1})

                # Set gamma to 0 if the batch is a terminal batch
                gam = gamma * (1 - np.array(terminal)).reshape(batch_size, 1)
                targets = categorical_loss_target(rewards, p_a_star, gam, cat.z.T, cat.dz, v_min, v_max)

                _, loss = sess.run([train_step, cat.loss],
                                   feed_dict={cat.target_probs: targets,
                                              cat.input_layer: S0,
                                              cat.actions: actions})

                print 'Step %s - Loss: %s - Buffer Size: %s - Eps: %s - Action: %s - Batch Rewards: %s' % (
                    step, loss, history.total_count, eps, action, np.sum(rewards))

                if target_update_mode == 'ewma':
                    sess.run(assign_ops)
                elif step % target_update_rate == 0:
                    sess.run(assign_ops)

        print 'Total Reward: %s' % total_reward
        episodes += 1

        if episodes % eval_interval == 0:
            eval_runner.run_episode(max_frames=3 * episodes / history.total_count)
