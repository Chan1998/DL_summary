import tensorflow as tf

import numpy as np

import collections

import random

import tensorflow.contrib.layers as layers

layers_list = [200,64,32,200]

MEMORY_SIZE = 2000


##built class for the DQN

class DeepQNetwork():

    def __init__(self, env, sess=None, gamma=0.8, epsilon=0.8, output_graph=False):
        self.gamma = gamma
        self.epsilon = epsilon

        # for gym:
        # self.action_dim = env.action_space.n
        # # self.action_dim = env.action_space.shape[0] #动作连续时
        # self.state_dim = env.observation_space.shape[0]
        # # self.state_dim = env.observation_space.n

        # for DSA:
        self.action_dim = env.action_space
        self.state_dim = env.observation_space


        self.network()
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

        if output_graph: # 是否使用tensorboard记录
            self.merged = tf.summary.merge_all()
            self.write = tf.summary.FileWriter("DQN/summaries", sess.graph)

        # memory for momery replay
        self.memory = []
        self.Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    # net_frame using for creating Q & target network

    def net_frame(self, hiddens, inpt, num_actions, scope, reuse=None):

        with tf.variable_scope(scope, reuse=False):
            out = inpt
            for hidden in hiddens:
                out = layers.fully_connected(out, num_outputs=hidden, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
            return out



    # create q_network & target_network

    def network(self):
        # q_network
        self.inputs_q = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim], name="inputs_q")
        scope_var = "DQN_q_network"
        self.q_value = self.net_frame(layers_list, self.inputs_q, self.action_dim, scope_var, reuse=True)

        # target_network

        self.inputs_target = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim], name="inputs_target")
        scope_tar = "DQN_target_network"
        self.q_target = self.net_frame(layers_list, self.inputs_target, self.action_dim, scope_tar)


        with tf.variable_scope("loss"):

            self.action = tf.placeholder(dtype=tf.int32, shape=[None], name="action")
            with tf.device('/cpu:0'):
                action_one_hot = tf.one_hot(self.action, self.action_dim)
            q_action = tf.reduce_sum(tf.multiply(self.q_value, action_one_hot), axis=1)


            self.target = tf.placeholder(dtype=tf.float32, shape=[None], name="target")
            self.loss = tf.reduce_mean(tf.square(q_action - self.target))
            # tf.summary.scalar('loss', tf.reduce_mean(self.loss))


        with tf.variable_scope("train"):
            optimizer = tf.train.AdamOptimizer(0.0001)
            self.train_op = optimizer.minimize(self.loss)

            # training

    def train(self, state, reward, action, state_next, done):
        q, q_target = self.sess.run([self.q_value, self.q_target],
                                    feed_dict={self.inputs_q: state, self.inputs_target: state_next})
        q_target_best = np.max(q_target, axis=1)
        q_target_best_mask = (1.0 - done) * q_target_best
        target = reward + self.gamma * q_target_best_mask
        # summery, loss, _ = self.sess.run([self.merged, self.loss, self.train_op],
        #
        #                                  feed_dict={self.inputs_q: state, self.target: target, self.action: action})
        # return summery, loss
        loss, _ = self.sess.run([self.loss, self.train_op],
                                         feed_dict={self.inputs_q: state, self.target: target, self.action: action})
        return loss
        # chose action



    def chose_action(self, current_state, test = False):
        current_state = current_state[np.newaxis, :]  # *** array dim: (xx,)  --> (1 , xx) ***
        q = self.sess.run(self.q_value, feed_dict={self.inputs_q: current_state})

        if test:
            return np.argmax(q)

        # e-greedy
        if np.random.random() < self.epsilon:
            action_chosen = np.random.randint(0, self.action_dim)
        else:
            action_chosen = np.argmax(q)

        return action_chosen



    # upadate parmerters

    def update_prmt(self):
        q_prmts = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "q_network")
        target_prmts = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "target_network")
        self.sess.run([tf.assign(t, q) for t, q in zip(target_prmts, q_prmts)])  # ***
        # print("updating target-network parmeters...")



    def decay_epsilon(self):
        if self.epsilon > 0.03:
            self.epsilon = self.epsilon - 0.02


    def add_memory(self, state, action, reward, next_state, done):      # 存储经验
        if len(self.memory) > MEMORY_SIZE:
            self.memory.pop(0)
        self.memory.append(self.Transition(state, action, reward, next_state, float(done)))

    def simple_memory(self, BATCH_SIZE):    # 批次取样提供训练数据
        # if len(self.memory) > BATCH_SIZE * 4:
        batch_transition = random.sample(self.memory, BATCH_SIZE)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = map(np.array,
                                                                                        zip(*batch_transition))
        return batch_state, batch_action, batch_reward, batch_next_state, batch_done


