import tensorflow as tf
import numpy as np

np.random.seed(2)
tf.set_random_seed(2)  # reproducible

#####################  hyper parameters  ####################
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_SIZE = 10000
BATCH_SIZE = 32

###############################  DDPG  ####################################
class DDPG(object):
    def __init__(self, env, sess=None):

        # self.s_dim = env.observation_space.shape[0]
        # self.a_dim = env.action_space.shape[0]
        # self.a_bound = env.action_space.high

        self.s_dim = env.observation_space
        self.a_dim = env.action_space
        self.a_bound = 2

        self.memory = np.zeros((MEMORY_SIZE, self.s_dim * 2 + self.a_dim + 1), dtype=np.float32)  #改动了，多减一
        self.pointer = 0

        self.sess = sess
        self.S = tf.placeholder(tf.float32, [None, self.s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, self.s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.a = self._build_a(self.S, )
        q = self._build_c(self.S, self.a, )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='DDPG_Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='DDPG_Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]  # soft update operation
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)  # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        self.a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(self.a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):    # soft replacement happened at here
            q_target = self.R + GAMMA * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        action = self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]
        # print(action)
        return action            #这时候action是一个两列的列表，对应不同动作概率。

    def learn(self):
        indices = np.random.choice(MEMORY_SIZE, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]  #改动了减一
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        loss = self.sess.run([self.a_loss, self.ctrain], {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
        return loss

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_SIZE  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('DDPG_Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.softmax, name='a', trainable=trainable)
            # return tf.multiply(a, self.a_bound, name='scaled_a')
            return a

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('DDPG_Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)
