import gym
import numpy as np

import cv2

GAMES = [
    'air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis',
    'bank_heist', 'battle_zone', 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout', 'carnival',
    'centipede', 'chopper_command', 'crazy_climber', 'demon_attack', 'double_dunk',
    'elevator_action', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar',
    'hero', 'ice_hockey', 'jamesbond', 'journey_escape', 'kangaroo', 'krull', 'kung_fu_master',
    'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong', 'pooyan',
    'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing',
    'solaris', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down',
    'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon'
]


class MaxAndSkipEnv(gym.Wrapper):
    """
    This class is copied from https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
    """
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype='uint8')
        self._skip = skip

    def _step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info


def process_frame(frame, shape=(84, 84, 1)):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, shape[:2], interpolation=cv2.INTER_AREA)
    return frame.reshape(shape) / 255.


class AtariEnvironment(object):
    def __init__(self, game='breakout', frameskip=4, max_random_start=30, fire_start=False, render=False):

        if game not in GAMES:
            raise ValueError('Invalid game {game}'.format(game=game))

        self.game = game
        self.frameskip = frameskip
        self.max_random_start = max_random_start
        self.render = render
        self.fire_start = fire_start

        name = ''.join([g.capitalize() for g in game.split('_')])

        name = '{name}NoFrameskip-v4'.format(name=name)

        env = gym.make(name)
        self.env = MaxAndSkipEnv(env, skip=self.frameskip)

        self.state_history = []
        self.reward_history = []
        self.terminal_history = []
        self.action_history = []
        self.info_history = []

    @property
    def action_space(self):
        return self.env.action_space

    def get_action_meanings(self):
        return self.env.env.env.get_action_meanings()

    def reset(self):
        state = self.env.reset()

        self.state_history = [state]
        self.action_history = []
        self.reward_history = []
        self.info_history = []

        return process_frame(state)

    def step(self, action):
        if self.render:
            self.env.render()

        state, reward, terminal, info = self.env.step(action)

        self.state_history.append(state)
        self.reward_history.append(reward)
        self.terminal_history.append(terminal)
        self.info_history.append(info)

        return process_frame(state), reward, terminal, info

    def random_start(self):
        state = self.reset()
        reward = 0
        terminal = False
        info = {}

        n_skip = np.random.randint(1, self.max_random_start + 1)

        actions = self.get_action_meanings()
        if 'FIRE' in actions and self.fire_start:
            action = actions.index('FIRE')
        else:
            action = actions.index('NOOP')

        for _ in xrange(n_skip):
            state, reward, terminal, info = self.step(action)

            if terminal:
                break

        return state, reward, terminal, info

    def get_state(self):
        return (process_frame(self.state_history[-1]), self.reward_history[-1],
                self.terminal_history[-1], self.info_history[-1])

    def close(self):
        self.env.env.close()
        self.env.close()
