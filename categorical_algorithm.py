import tensorflow as tf
import numpy as np


class CategoricalAlgorithm(object):
    def __init__(self, config, n_actions, name=None):
        self.config = config
        n_atoms = self.config['n_atoms']
        self.n_actions = n_actions

        if name:
            self.name = name
        else:
            self.name = 'categorical%s' % self.config['n_atoms']

        v_min, v_max = self.config['v_bounds']

        self.dz = (v_max - v_min) / float(n_atoms - 1)
        self.z = (v_min + np.arange(n_atoms).reshape((n_atoms, 1)) * self.dz).astype(np.float32)

        self.__initialize_model()

    def __initialize_model(self):
        input_size = [None] + list(self.config['input_size'])
        input_size[-1] = input_size[-1] * self.config['state_history']

        self.input_layer = tf.placeholder(tf.float32, input_size, name='%s/input_layer' % self.name)
        self.temperature = tf.placeholder(tf.float32, (), name='%s/temperature' % self.name)
        self.actions = tf.placeholder(tf.int32, [None, 2], name='%s/actions' % self.name)
        self.target_probs = tf.placeholder(tf.float32, [None, self.config['n_atoms']],
                                           name='%s/target_probs' % self.name)

        self.layers = [self.input_layer]
        for i, (f, k, s, p) in enumerate(zip(self.config['filters'],
                                             self.config['kernels'],
                                             self.config['strides'],
                                             self.config.get('padding', ['valid'] * len(self.config['filters'])))):
            conv = tf.layers.conv2d(
                inputs=self.layers[-1],
                filters=f,
                kernel_size=(k, k),
                strides=(s, s),
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                name='%s/conv%s' % (self.name, i)
            )
            self.layers.append(conv)

        flat = tf.reshape(self.layers[-1], [-1, np.prod([int(d) for d in self.layers[-1].shape[1:]])])
        self.layers.append(flat)

        for i, dim in enumerate(self.config['fc']):
            fc = tf.layers.dense(
                inputs=self.layers[-1],
                units=dim,
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                name='%s/dense%i' % (self.name, i),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.config.get('l2_reg', 0.0))
            )
            self.layers.append(fc)

        nwa = '%s/ouput_layer/Wcat' % self.name
        nba = '%s/ouput_layer/bcat' % self.name

        self.Wcat_actions = tf.get_variable(
            nwa,
            [self.config['fc'][-1], self.n_actions, self.config['n_atoms']],
            tf.float32,
            tf.contrib.layers.variance_scaling_initializer(),
            regularizer=tf.contrib.layers.l2_regularizer(scale=self.config.get('l2_reg', 0.0)))
        self.bcat_actions = tf.get_variable(
            nba,
            [1, self.n_actions, self.config['n_atoms']])

        with tf.name_scope('%s/action_logits' % self.name):
            self.action_logits = tf.tensordot(self.layers[-1], self.Wcat_actions, axes=[[1], [0]]) + self.bcat_actions

        self.logits = self.action_logits

        self.layers.append(self.logits)

        self.probs = tf.nn.softmax(self.logits, name='%s/action_softmax' % self.name)
        self.probs_with_temperature = tf.nn.softmax(self.logits / self.temperature,
                                                    name='%s/temperature_action_softmax' % self.name)

        with tf.name_scope('%s/Q' % self.name):
            self.Q = tf.reshape(tf.tensordot(self.probs, self.z, axes=[[2], [0]]), [-1, self.n_actions])

        if dueling:
            with tf.name_scope('%s/advantage' % self.name):
                self.A = self.Q - self.V

        self.loss = self._loss()

    def _loss(self):
        with tf.name_scope('%s/loss' % self.name):
            logits = tf.gather_nd(self.logits, self.actions)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.target_probs)
            cross_entropy_loss = tf.reduce_mean(cross_entropy, name='cross_entropy_loss')

        with tf.name_scope('%s/l2_loss' % self.name):
            l2_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        final_loss = cross_entropy_loss + l2_loss

        return final_loss

    def q_values(self, X, session):
        return session.run(self.Q, feed_dict={self.input_layer: X})

    def get_actions(self, X, session):
        q = self.q_values(X, session)
        return q.argmax(1)


def categorical_loss_target(rewards, target_probs, gamma, z, dz, v_min, v_max):
    Tz = np.clip(rewards + gamma * z, v_min, v_max)
    b = (Tz - v_min) / dz
    l = np.floor(b).astype(int)
    u = np.clip(np.floor(b + 1), 0, target_probs.shape[1] - 1).astype(int)
    m = np.zeros_like(Tz)
    t_umb = target_probs * (u - b)
    t_bml = target_probs * (b - l)
    for i in xrange(target_probs.shape[0]):
        for j in xrange(target_probs.shape[1]):
            m[i, l[i, j]] += t_umb[i, j]
            m[i, u[i, j]] += t_bml[i, j]

    return m

