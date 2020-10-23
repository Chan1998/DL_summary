import numpy as np

import tensorflow as tf

# import gym

# import os
# os.environ[ "CUDA_VISIBLE_DEVICES"] = "-1"
#
# # os.environ[ "CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


#import matplotlib.pyplot as plt
# import seaborn as sns

import matplotlib.pyplot as plt



np.random.seed(2)
tf.set_random_seed(2)  # reproducible



# 超参数

# MAX_EPISODE = 300
#
# DISPLAY_REWARD_THRESHOLD = 200  # 刷新阈值
#
# MAX_STEPS = 10000 # 最大迭代次数
#
# RENDER = True  # 渲染开关

# GAMMA = 0.9  # 衰变值

LR_A = 0.001  # Actor学习率

LR_C = 0.001  # Critic学习率



# env = gym.make('CartPole-v0')
# env = gym.make('MountainCar-v0')
# env = gym.make('Acrobot-v1')
# env.seed(1)

# env = env.unwrapped



class Actor(object):

    def __init__(self, env, sess=None, lr=LR_A, output_graph=False):

        self.sess = sess
        # for gym:
        # self.n_features = env.observation_space.shape[0]
        # self.n_actions = env.action_space.n

        #for DSA:
        self.n_features = env.observation_space
        self.n_actions = env.action_space


        self.s = tf.placeholder(tf.float32, [1, self.n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error



        with tf.variable_scope('AC_Actor'):

            l1 = tf.layers.dense(
                inputs=self.s,
                units=200,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            # l2 = tf.layers.dense(
            #     inputs=l1,
            #     units=200,  # number of hidden units
            #     activation=tf.nn.relu,
            #     kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            #     bias_initializer=tf.constant_initializer(0.1),  # biases
            #     name='l2'
            # )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=self.n_actions,  # output units
                activation=tf.nn.softmax,  # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):

            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

            # tf.summary.scalar('exp_v', tf.reduce_mean(self.exp_v))

        with tf.variable_scope('train'):

            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)


        sess.run(tf.global_variables_initializer())  # 初始化参数

        if output_graph:
            self.merged = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES,'exp_v'))

    def learn(self, s, a, td):

        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}

        # summery, exp_v, _= self.sess.run([self.merged, self.exp_v, self.train_op], feed_dict)
        #
        # return exp_v, summery

        exp_v, _ = self.sess.run([self.exp_v, self.train_op], feed_dict)

        return exp_v


    def chose_action(self, s, test=False):

        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})  # 获取所有操作的概率

        if test:
            return np.argmax(probs)

        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())  # return a int

class Critic(object):

    def __init__(self, env, sess=None, lr=LR_C, gamma=0.9, output_graph=False):
        self.sess = sess
        # self.n_features = env.observation_space.shape[0]

        self.n_features = env.observation_space
        self.s = tf.placeholder(tf.float32, [1, self.n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')


        with tf.variable_scope('AC_Critic'):

            l1 = tf.layers.dense(
                inputs=self.s,
                units=200,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            # l2 = tf.layers.dense(
            #     inputs=l1,
            #     units=200,  # number of hidden units
            #     activation=tf.nn.relu,  # None
            #     # have to be linear to make sure the convergence of actor.
            #     # But linear approximator seems hardly learns the correct Q.
            #     kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            #     bias_initializer=tf.constant_initializer(0.1),  # biases
            #     name='l2'
            # )



            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )



        with tf.variable_scope('squared_TD_error'):

            self.td_error = self.r + gamma * self.v_ - self.v
            self.loss = tf.square(self.td_error)  # TD_error = (r+gamma*V_next) - V_eval
            # tf.summary.scalar('loss', tf.reduce_mean(self.loss))

        with tf.variable_scope('train'):

            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)
        sess.run(tf.global_variables_initializer())  # 初始化参数

        if output_graph:
            self.merged = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES,'squared_TD_error') )

    def learn(self, s, r, s_):

        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        v_ = self.sess.run(self.v, {self.s: s_})

        # td_error, summery,  _ = self.sess.run([self.td_error, self.merged, self.train_op],
        #
        #                             {self.v_: v_, self.r: r, self.s: s})
        #
        # return td_error, summery

        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                    {self.v_: v_, self.r: r, self.s: s})

        return td_error


