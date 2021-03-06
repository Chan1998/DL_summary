import tensorflow as tf
from visualdl import LogWriter
from PIL import Image
import numpy as np
import random
import seaborn as sns
from tqdm import tqdm
# import gym
import matplotlib.pyplot as plt
import DQN_agent as DQN
import Actor_Critic_agent as AC
# import PolicyGradient_agent as PG
import DDPG_agent as ddpg
import DRQN_agent as drqn
import DCQN_agent as dcqn
import DSA_env as ENV

# import os
# # os.environ[ "CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# os.environ[ "CUDA_VISIBLE_DEVICES"] = "-1"

# ENV = "CartPole-v0" #可用
# ENV = "MountainCar-v0" #可用
# ENV = "Acrobot-v1"   #可用
# ENV = "CarRacing-v0"
# ENV = "Pendulum-v0" #连续动作比较复杂，DQN不行
# ENV ='KellyCoinflip-v0' #状态比较复杂

MEMORY_SIZE = 2000
EPISODES = 200
MAX_STEP = 500  # 注意小于state总时隙数
BATCH_SIZE = 32
UPDATE_PERIOD = 200  # update target network parameters
EXPLOR_PERIOD = (0.6 * (EPISODES * MAX_STEP) // 40)
# print(EXPLOR_PERIOD)
# SHOW_PERIOD = 400
# layers_list = [200,64]



def random_chose(env):
    print("******************开始随机对比*********************")
    # reward_list = []
    reward_list_epsiod = []
    reward_list = []
    do_num_list = []


    # 开始训练
    for episode in tqdm(range(EPISODES)):
        state = env.reset(random_start=random_start_list[episode])
        reward_all = 0
        do_num = 0
        action_list = []
        # training
        for step in range(MAX_STEP):
            # if episode % 5 == 1:
            #     env.render()
            action = random.randint(0, env.action_space - 1)
            if action:
                do_num += 1
            action_list.append(action)
            _, reward, _  = env.step(action, step, random_start=random_start_list[episode])
            reward_all += reward
            # reward_list.append(float(reward_all)/float(step + 1))
            if episode == (EPISODES - 1):
                reward_list.append(reward_all)
        reward_list_epsiod.append(reward_all)
        # print("step = {} action = {} result = {} [reward_all = {}] [success_rate = {}]".format(step, action,
        #                                                                                        state[-2] != action,
        #                                                                                        reward_all, float(
        #         reward_all) / float(step + 1)))
    return action_list, reward_list, reward_list_epsiod, do_num_list


def Train_DQN(env, agent):

    print("******************开始DQN训练*********************")

    update_iter = 0
    step_iter = 0
    # reward_list = []
    reward_list_epsiod = []
    do_num_list = []

    loss_list = []
    reward_list = []
    # 开始训练
    with LogWriter(logdir="./log/train/DQN") as writer:
        for episode in tqdm(range(EPISODES)):
            state = env.reset(random_start=random_start_list[episode])
            state = state.reshape(env.channel_num * env.time_step)
            reward_all = 0
            do_num = 0
            action_list = []
            # training
            # for step in tqdm(range(MAX_STEP)):
            for step in range(MAX_STEP):
                # if episode % 5 == 1:
                #     env.render()
                action = agent.chose_action(state)
                if action:
                    do_num += 1
                next_state, reward ,done = env.step(action, step, random_start=random_start_list[episode])
                next_state = next_state.reshape(env.channel_num * env.time_step)

                reward_all += reward
                action_list.append(action)
                if episode == (EPISODES - 1):
                    reward_list.append(reward_all)
                    writer.add_scalar(tag="action/last_episods", step=step, value=action)
                    writer.add_scalar(tag="reward/reward", step=step, value=reward)


                  # 存储
                agent.add_memory(state, action, reward, next_state, done)

                if len(agent.memory) > BATCH_SIZE * 2:  # 采样
                    batch_state, batch_action, batch_reward, batch_next_state, batch_done = agent.simple_memory(BATCH_SIZE)

                    loss = agent.train(state=batch_state,  # 训练
                                     reward=batch_reward,
                                     action=batch_action,
                                     state_next=batch_next_state,
                                     done=batch_done
                                     )
                    # DQN.write.add_summary(summery, update_iter)  #记录损失数据
                    update_iter += 1
                    writer.add_scalar(tag="loss", step=update_iter, value=loss)

                    loss_list.append(loss)  # 用于绘图

                if update_iter % UPDATE_PERIOD == 1:  # 更新target网络
                    agent.update_prmt()

                if update_iter % EXPLOR_PERIOD == 1:  # 减小探索概率
                    agent.decay_epsilon()
                    # writer.add_scalar(tag="epsilon_probability", step=update_iter, value=agent.epsilon)

                # if update_iter % SHOW_PERIOD == 1:  # 更新target网络
                #     print("epsiods = {}, step = {} loss = {} action = {} result = {} [reward_all = {}] [success_rate = {}]".format(episode, step, loss, action, reward, reward_all, float(reward_all)/float(step + 1)))

                state = next_state
                # step_iter += 1

            writer.add_histogram(tag='action/action_list', values=action_list, step=episode, buckets=env.action_space)
            reward_list_epsiod.append(reward_all)
            writer.add_scalar(tag="reward/episode", step=episode, value=reward_all)
            do_num_list.append(do_num)
            writer.add_scalar(tag="action/episode", step=episode, value=do_num)
            # print(
            #     "epsiods = {} epsilon = {} loss = {} action = {} result = {} [reward_all = {}] [success_rate = {}]".format(
            #         episode, agent.epsilon, loss, action, reward, reward_all, float(reward_all) / float(step + 1)))

    return action_list, reward_list, loss_list, reward_list_epsiod, do_num_list


def Train_DRQN(env, agent):
    print("******************开始DRQN训练*********************")

    update_iter = 0
    # step_iter = 0
    # reward_list = []
    reward_list_epsiod = []
    do_num_list = []
    loss_list = []
    reward_list = []
    # 开始训练
    with LogWriter(logdir="./log/train/DRQN") as writer:
        for episode in tqdm(range(EPISODES)):
            state = env.reset(random_start=random_start_list[episode])
            state = state.reshape(env.channel_num * env.time_step)
            reward_all = 0
            do_num = 0
            action_list = []
            # training
            # for step in tqdm(range(MAX_STEP)):
            for step in range(MAX_STEP):
                # if episode % 5 == 1:
                #     env.render()
                action = agent.chose_action(state)
                if action:
                    do_num += 1
                next_state, reward, done = env.step(action, step, random_start=random_start_list[episode])
                next_state = next_state.reshape(env.channel_num * env.time_step)

                reward_all += reward
                action_list.append(action)
                if episode == (EPISODES - 1):
                    reward_list.append(reward_all)
                    writer.add_scalar(tag="action/last_episods", step=step, value=action)
                    writer.add_scalar(tag="reward/reward", step=step, value=reward)

                # 存储
                agent.add_memory(state, action, reward, next_state, done)

                if len(agent.memory) > BATCH_SIZE * 2:  # 采样
                    batch_state, batch_action, batch_reward, batch_next_state, batch_done = agent.simple_memory(BATCH_SIZE)

                    loss = agent.train(state=batch_state,  # 训练
                                       reward=batch_reward,
                                       action=batch_action,
                                       state_next=batch_next_state,
                                       done=batch_done
                                       )
                    # DQN.write.add_summary(summery, update_iter)  #记录损失数据
                    update_iter += 1
                    writer.add_scalar(tag="loss", step=update_iter, value=loss)
                    loss_list.append(loss)  # 用于绘图

                if update_iter % UPDATE_PERIOD == 1:  # 更新target网络
                    agent.update_prmt()

                if update_iter % EXPLOR_PERIOD == 1:  # 减小探索概率
                    agent.decay_epsilon()
                    # writer.add_scalar(tag="epsilon_probability", step=update_iter, value=agent.epsilon)

                # if update_iter % SHOW_PERIOD == 1:
                #     print(
                #         "epsiods = {}, step = {} loss = {} action = {} result = {} [reward_all = {}]
                #         [success_rate = {}]".format(
                #             episode, step, loss, action, reward, reward_all, float(reward_all) / float(step + 1)))

                state = next_state
                # step_iter += 1

            writer.add_histogram(tag='action/action_list', values=action_list, step=episode, buckets=env.action_space)
            reward_list_epsiod.append(reward_all)
            writer.add_scalar(tag="reward/episode", step=episode, value=reward_all)
            do_num_list.append(do_num)
            writer.add_scalar(tag="action/episode", step=episode, value=do_num)

            # print(
            #     "epsiods = {} epsilon = {} loss = {} action = {} result = {} [reward_all = {}]
            #     [success_rate = {}]".format(
            #         episode, agent.epsilon, loss, action, reward, reward_all, float(reward_all) / float(step + 1)))

    return action_list, reward_list, loss_list, reward_list_epsiod, do_num_list


def Train_DCQN(env, agent):

    print("******************开始DCQN训练*********************")

    update_iter = 0
    # step_iter = 0
    # reward_list = []
    reward_list_epsiod = []
    do_num_list = []


    loss_list = []
    reward_list = []
    # 开始训练
    with LogWriter(logdir="./log/train/DCQN") as writer:
        for episode in tqdm(range(EPISODES)):
            state = env.reset(random_start=random_start_list[episode])
            reward_all = 0
            do_num = 0
            action_list = []
            # training
            # for step in tqdm(range(MAX_STEP)):
            for step in range(MAX_STEP):
                # if episode % 5 == 1:
                #     env.render()
                action = agent.chose_action(state)
                if action:
                    do_num += 1
                next_state, reward, done = env.step(action, step, random_start=random_start_list[episode])
                reward_all += reward
                action_list.append(action)
                if episode == (EPISODES-1):
                    reward_list.append(reward_all)
                    writer.add_scalar(tag="action/last_episods", step=step, value=action)
                    writer.add_scalar(tag="reward/reward", step=step, value=reward)

                  # 存储
                agent.add_memory(state, action, reward, next_state, done)

                if len(agent.memory) > BATCH_SIZE * 2:  # 采样
                    batch_state, batch_action, batch_reward, batch_next_state, batch_done = agent.simple_memory(BATCH_SIZE)

                    loss = agent.train(state=batch_state,  # 训练
                                     reward=batch_reward,
                                     action=batch_action,
                                     state_next=batch_next_state,
                                     done=batch_done
                                     )
                    # DQN.write.add_summary(summery, update_iter)  #记录损失数据
                    update_iter += 1
                    writer.add_scalar(tag="loss", step=update_iter, value=loss)

                    loss_list.append(loss)  # 用于绘图

                if update_iter % UPDATE_PERIOD == 1:  # 更新target网络
                    agent.update_prmt()

                if update_iter % EXPLOR_PERIOD == 1:  # 减小探索概率
                    agent.decay_epsilon()
                    # writer.add_scalar(tag="epsilon_probability", step=update_iter, value=agent.epsilon)

                # if update_iter % SHOW_PERIOD == 1:
                #     print("epsiods = {}, step = {} loss = {} action = {} result = {} [reward_all = {}] [success_rate = {}]".format(episode, step, loss, action, reward, reward_all, float(reward_all)/float(step + 1)))

                state = next_state
                # step_iter += 1

            writer.add_histogram(tag='action/action_list', values=action_list, step=episode, buckets=env.action_space)
            reward_list_epsiod.append(reward_all)
            writer.add_scalar(tag="reward/episode", step=episode, value=reward_all)
            do_num_list.append(do_num)
            writer.add_scalar(tag="action/episode", step=episode, value=do_num)

            # print(
            #     "epsiods = {} epsilon = {} loss = {} action = {} result = {} [reward_all = {}] [success_rate = {}]".format(
            #         episode, agent.epsilon ,loss, action, reward, reward_all, float(reward_all) / float(step + 1)))

    return action_list, reward_list, loss_list, reward_list_epsiod, do_num_list

# def Test_DQN(env, agent):
#     print("*******************开始测试**************")
#
#     state = env.reset()
#     reward_all = 0
#     # training
#     for step in range(MAX_STEP):
#         env.render()
#         action = agent.chose_action(state, test=True)
#         next_state, reward, done, _ = env.step(action)
#
#         # for cartbar
#         # x 是车的水平位移, 所以 r1 是车越偏离中心, 分越少
#         # theta 是棒子离垂直的角度, 角度越大, 越不垂直. 所以 r2 是棒越垂直, 分越高
#         x, x_dot, theta, theta_dot = next_state  # 细分开, 为了修改原配的 reward
#         r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
#         r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
#         reward = r1 + r2  # 总 reward 是 r1 和 r2 的结合, 既考虑位置, 也考虑角度, 这样 DQN 学习更有效率
#
#         # # for montaincar
#         # # # 车开得越高 reward 越大
#         # position, velocity = next_state
#         # reward = abs(position - (-0.5))
#
#         reward_all += reward
#
#         if done:
#             print("step = {} [reward_all = {}]".format(step, reward_all))
#             # reward_list.append(step)
#             break
#
#         state = next_state

# def Train_PG(env, agent):
#     print("******************开始P_G训练*********************")
#
#     update_iter = 0
#     reward_list = []
#     loss_list = []
#     # 开始训练
#     for episode in range(EPISODES):
#         state = env.reset()
#         reward_all = 0
#         # training
#         for step in range(MAX_STEP):
#             # if episode % 5 == 1:
#             #     env.render()
#             action = agent.choose_action(state)
#             next_state, reward, done, _ = env.step(action)
#
#
#             # for cartbar
#             # x 是车的水平位移, 所以 r1 是车越偏离中心, 分越少
#             # theta 是棒子离垂直的角度, 角度越大, 越不垂直. 所以 r2 是棒越垂直, 分越高
#             # x, x_dot, theta, theta_dot = next_state  # 细分开, 为了修改原配的 reward
#             # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
#             # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
#             # reward = r1 + r2  # 总 reward 是 r1 和 r2 的结合, 既考虑位置, 也考虑角度, 这样 DQN 学习更有效率
#
#             # # for montaincar
#             # # # 车开得越高 reward 越大
#             # position, velocity = next_state
#             # reward = abs(position - (-0.5))
#
#             agent.store_transition(state, action, float(reward))
#             reward_all += reward
#             reward_list.append(float(reward_all)/float(step + 1))
#
#             # if done:
#                 # if episode % 10 == 1:
#                 #     print("episode:", episode, "  step:", step, "  reward:", reward_all)
#                 # reward_list.append(step)
#             if step % UPDATE_PERIOD == 5:
#                 loss, _ = agent.learn()
#                 loss_list.append(loss)  # 用于绘图
#                 update_iter += 1
#                 # break
#             # print("step = {} action = {} result = {} [reward_all = {}] [success_rate = {}]".format(step, action,
#             #                                                                                        state[-2] != action,
#             #                                                                                        reward_all, float(
#             #         reward_all) / float(step + 1)))
#             state = next_state
#     return reward_list, loss_list

def Train_AC(env, actor, critic):

    print("******************开始A_C训练*********************")

    update_iter = 0
    # reward_list = []
    reward_list_epsiod = []

    loss_list = []
    # 开始训练
    for episode in tqdm(range(EPISODES)):
        state = env.reset()
        state = state.reshape(env.channel_num * env.time_step)
        reward_all = 0
        reward_list = []
        action_list = []
        # training
        for step in tqdm(range(MAX_STEP)):
            # if episode % 5 == 1:
            #     env.render()
            action = actor.chose_action(state)
            # print(action)
            action_list.append(action)
            next_state, reward, done = env.step(action, step)
            next_state = next_state.reshape(env.channel_num * env.time_step)

            reward_all += reward
            # reward_list.append(float(reward_all)/float(step + 1))
            reward_list.append(reward_all)

            td_error = critic.learn(state, reward, next_state)  # Critic 学习
            loss_list.append(abs(td_error[0][0]))  # 记录回报值r
            actor.learn(state, action, td_error)  # Actor 学习
            # _, summery = actor.learn(state, action, td_error)  # Actor 学习
            # write.add_summary(summery, update_iter)
            actor.learn(state, action, td_error)  # Actor 学习
            # if update_iter % SHOW_PERIOD == 1:  # 更新target网络
            #     print(
            #         "epsiods = {}, step = {} loss = {} action = {} result = {} [reward_all = {}] [success_rate = {}]".format(
            #             episode, step, abs(td_error[0][0]), action, reward, reward_all, float(reward_all) / float(step + 1)))

            update_iter += 1

            # if done:
            #     if episode % 10 == 1:
            #         print("episode = {}  step = {} [reward_all = {}]".format(episode, step, reward_all))
            #     reward_list.append(step)
            #     break

            # print("step = {} action = {} result = {} [reward_all = {}] [success_rate = {}]".format(step, action,
            #                                                                                        state[-2] != action,
            #                                                                                        reward_all, float(
            #         reward_all) / float(step + 1)))
            state = next_state

        reward_list_epsiod.append(reward_all)

        print(
            "epsiods = {} loss = {} action = {} result = {} [reward_all = {}] [success_rate = {}]".format(
                episode, td_error[0][0], action, reward, reward_all, float(reward_all) / float(step + 1)))

    return action_list, reward_list, loss_list, reward_list_epsiod

# def Test_AC(env, actor):
#     print("*******************开始测试**************")
#
#     state = env.reset()
#     reward_all = 0
#     # training
#     for step in range(MAX_STEP):
#         env.render()
#         action = actor.chose_action(state, test=True)
#         next_state, reward, done, _ = env.step(action)
#
#         # for cartbar
#         # x 是车的水平位移, 所以 r1 是车越偏离中心, 分越少
#         # theta 是棒子离垂直的角度, 角度越大, 越不垂直. 所以 r2 是棒越垂直, 分越高
#         x, x_dot, theta, theta_dot = next_state  # 细分开, 为了修改原配的 reward
#         r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
#         r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
#         reward = r1 + r2  # 总 reward 是 r1 和 r2 的结合, 既考虑位置, 也考虑角度, 这样 DQN 学习更有效率
#
#         # # for montaincar
#         # # # 车开得越高 reward 越大
#         # position, velocity = next_state
#         # reward = abs(position - (-0.5))
#
#         reward_all += reward
#
#         if done:
#             print("step = {} [reward_all = {}]".format(step, reward_all))
#             # reward_list.append(step)
#             break
#
#         state = next_state

def Train_DDPG(env, agent):
    print("******************开始DDPG训练*********************")

    # update_iter = 0
    reward_list = []
    action_list = []
    loss_list = []
    # 开始训练
    var = 3  # control exploration
    for episode in range(EPISODES):
        state = env.reset()
        reward_all = 0
        # training
        for step in range(MAX_STEP):
            # if episode % 20 == 1:
            #     env.render()
            action_unit = agent.choose_action(state)
            # action = np.clip(np.random.normal(action, var), -2, 2)  # add randomness to action selection for exploration
            #对动作列表进行处理
            action = np.random.choice(np.arange(action_unit.shape[0]), p=action_unit.ravel())
            action_list.append(action)
            next_state, reward, done, _ = env.step(action)

            # for cartbar
            # x 是车的水平位移, 所以 r1 是车越偏离中心, 分越少
            # theta 是棒子离垂直的角度, 角度越大, 越不垂直. 所以 r2 是棒越垂直, 分越高
            # x, x_dot, theta, theta_dot = next_state  # 细分开, 为了修改原配的 reward
            # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            # reward = r1 + r2  # 总 reward 是 r1 和 r2 的结合, 既考虑位置, 也考虑角度, 这样 DQN 学习更有效率

            # # for montaincar
            # # # 车开得越高 reward 越大
            # position, velocity = next_state
            # reward = abs(position - (-0.5))

            reward_all += reward
            reward_list.append(float(reward_all) / float(step + 1))
            agent.store_transition(state, action_unit, reward, next_state)

            if agent.pointer > MEMORY_SIZE:
                var *= .9995  # decay the action randomness
                loss = agent.learn()
                loss_list.append(loss)

            # if done:
            #     if episode % 10 == 1:
            #         print("episode = {}  step = {} [reward_all = {}]".format(episode, step, reward_all))
            #     reward_list.append(reward_all)
            #     break

            state = next_state

            # print("step = {} action = {} result = {} [reward_all = {}] [success_rate = {}]".format(step, action, state[-2]!=action, reward_all, float(reward_all)/float(step + 1)))
    return action_list, reward_list, loss_list


if __name__ == "__main__":

    # env = gym.make(ENV)
    # env = env.unwrapped

    # print(env.action_space)
    # print(env.observation_space)

    random_start_list = np.random.randint(0,10000-MAX_STEP,EPISODES)
    # print(random_start_list)
    # print(np.shape(random_start_list))

    env = ENV.DSA()

    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7  # 固定比例占用显存
    with tf.Session(config=config) as sess:
        # P_G = PG.PolicyGradient(env, sess)
        # reward_list_PG, loss_list_PG = Train_PG(env, P_G)

        # action_list_random, reward_list_random, episod_reward_list_random, do_num_Random = random_chose(env)

        DCQN = dcqn.DeepQNetwork(env, sess)
        action_list_DCQN, reward_list_DCQN, loss_list_DCQN, episod_reward_list_DCQN, do_num_DCQN = Train_DCQN(env, DCQN)

        # actor = AC.Actor(env, sess)  # 初始化Actor
        # critic = AC.Critic(env, sess)  # 初始化Critic
        # action_list_AC, reward_list_AC, loss_list_AC, epsiod_reward_list_AC = Train_AC(env, actor, critic)

        DRQN = drqn.DeepQNetwork(env, sess)
        action_list_DRQN, reward_list_DRQN, loss_list_DRQN, episod_reward_list_DRQN, do_num_DRQN = Train_DRQN(env, DRQN)



        # # Test_AC(env, actor)

        # DDPG = ddpg.DDPG(env, sess)
        # action_list_DDPG, reward_list_DDPG, loss_list_DDPG = Train_DDPG(env, DDPG)

        DQN = DQN.DeepQNetwork(env, sess)
        action_list_DQN, reward_list_DQN, loss_list_DQN, episod_reward_list_DQN, do_num_DQN = Train_DQN(env, DQN)
        # Test_DQN(env, DQN)


        # # Test_DQN(env, DQN)

        plt.figure()
        plt.plot(
                    # reward_list_AC[5:], 'y-',
                    reward_list_DCQN[5:], 'c-',
                    reward_list_DRQN[5:], 'y-',
                    reward_list_DQN[5:], 'm-',
                    # reward_list_random[5:], 'r-'
                 )
        plt.xlabel("(steps)")
        plt.ylabel("success_rate")
        plt.title("success_rate")
        plt.legend(['DCQN','DRQN','DQN'])

        plt.figure()
        plt.plot(
                    # epsiod_reward_list_AC, 'y--',
                    episod_reward_list_DCQN, 'co-',
                    episod_reward_list_DRQN, 'y<-',
                    episod_reward_list_DQN, 'ms-',
                    # episod_reward_list_random, 'r--'
        )
        plt.xlabel("(episods)")
        plt.ylabel("success_number")
        plt.title("success_episod")
        plt.legend(['DCQN','DRQN','DQN'])

        plt.figure()
        plt.plot(
            # epsiod_reward_list_AC, 'y--',
            do_num_DCQN, 'co-',
            do_num_DRQN, 'y<-',
            do_num_DQN, 'ms-',
            # do_num_Random, 'r+-'
        )
        plt.xlabel("(episods)")
        plt.ylabel("do_number")
        plt.title("do_num_episod")
        plt.legend(['DCQN', 'DRQN', 'DQN'])

        # plt.figure()
        # plt.plot(loss_list_AC, 'r-')
        # plt.xlabel("(train_steps)")
        # plt.ylabel("loss")
        # plt.title("AC_loss")

        # plt.figure()
        # plt.plot(loss_list_DQN, 'r-')
        # plt.xlabel("(train_steps)")
        # plt.ylabel("loss")
        # plt.title("DQN_loss")
        #
        # plt.figure()
        # plt.plot(loss_list_DCQN, 'r-')
        # plt.xlabel("(train_steps)")
        # plt.ylabel("loss")
        # plt.title("DCQN_loss")
        #
        # plt.figure()
        # plt.plot(loss_list_DRQN, 'r-')
        # plt.xlabel("(train_steps)")
        # plt.ylabel("loss")
        # plt.title("DRQN_loss")
        #
        # plt.figure()
        # plt.plot(np.log(loss_list_DCQN), 'r-')
        # plt.xlabel("(train_steps)")
        # plt.ylabel("loss_(dB)")
        # plt.title("DCQN_loss_dB")
        #
        # # plt.figure()
        # # plt.plot(np.log(loss_list_AC), 'r-')
        # # plt.xlabel("(train_steps)")
        # # plt.ylabel("loss_(dB)")
        # # plt.title("AC_loss_dB")
        #
        # plt.figure()
        # plt.plot(np.log(loss_list_DRQN), 'r-')
        # plt.xlabel("(train_steps)")
        # plt.ylabel("loss_(dB)")
        # plt.title("DRQN_loss_dB")
        #
        #
        #
        # plt.figure()
        # plt.plot(np.log(loss_list_DQN), 'r-')
        # plt.xlabel("(train_steps)")
        # plt.ylabel("loss_(dB)")
        # plt.title("DQN_loss_dB")

        plt.figure()
        sum = np.array([
                        # action_list_AC[-90:],
                        action_list_DQN[-90:],
                        action_list_DCQN[-90:],
                        action_list_DRQN[-90:],
                        # action_list_random[-50:],
                        ]).reshape([3, -1])
        sns.heatmap(sum, cmap='Reds', yticklabels=['DQN', 'DCQN', 'DRQN'])
        plt.xlabel('time_slots')
        plt.ylabel('RL_agent_type')
        plt.title('agent_action(-50)')


        plt.show()
