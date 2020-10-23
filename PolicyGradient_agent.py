import tensorflow as tf
import numpy as np

np.random.seed(2)
tf.set_random_seed(2)  # reproducible

class PolicyGradient:
    def __init__(self, env, sess=None, learning_rate=0.02, reward_decay=0.99, output_graph=False):

        self.lr = learning_rate
        self.gamma = reward_decay

        #for gym
        # self.n_actions = env.action_space.n
        # # self.n_actions = env.action_space.shape[0] #动作连续时
        # self.n_features = env.observation_space.shape[0]
        # # self.n_features = env.observation_space.n

        # for DSA:
        self.n_actions = env.action_space
        self.n_features = env.observation_space

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self._build_net()

        self.sess = sess
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")

            # fc1
            layer1 = tf.layers.dense(
                inputs=self.tf_obs,
                units=200,
                activation=tf.nn.relu,  # relu activation
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                bias_initializer=tf.constant_initializer(0.1),
                name='fc1'
            )

            # fc2
            layer2 = tf.layers.dense(
                inputs=layer1,
                units=200,
                activation=tf.nn.relu,  # relu activation
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                bias_initializer=tf.constant_initializer(0.1),
                name='fc2'
            )

            # fc3
            all_act = tf.layers.dense(
                inputs=layer2,
                units=self.n_actions,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                bias_initializer=tf.constant_initializer(0.1),
                name='fc3'
            )

            self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability
            self.all_act_prob = tf.clip_by_value(self.all_act_prob, 1e-5,1-1e-5)

            with tf.name_scope('loss'):
                # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
                neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act,
                                                                              labels=self.tf_acts)  # this is negative log of chosen action
                # or in this way:
                # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)

                self.loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

            with tf.name_scope('train'):
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        # print(prob_weights)
        # prob_weights = prob_weights + 1e-6
        # print(prob_weights)
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        # action = np.argmax(prob_weights[0])
        # print(action)
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={
            self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
            self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
            self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # empty episode data
        return loss, discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            # print(running_add)
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs + 1e-6)
        return discounted_ep_rs