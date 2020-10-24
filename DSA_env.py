import random
import numpy as np
import collections
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt



#  PU占用模式：
# 只有一个信道，间歇占用[001001001001]

# 读取.文件
data_in = pd.read_csv("./channel_data/period_state71113.csv")

class DSA():
    def __init__(self):
        self.channel_num = 3  # 可用信道数
        self.time_step = 7   # 输入状态历史时间步数

        self.action_space = self.channel_num + 1  # (0表示不接入信道，其余表示接入对应信道)
        self.observation_space = self.channel_num * self.time_step  # [[state * self.time_step] * self.channel_num]每个信道时间步数 * 信道个数
        # self.observation_space = [self.channel_num , self.time_step]

        self.state = collections.deque(maxlen=self.channel_num)  # (channel_num, time_steps)

    def reset(self):
        # 情景一：假设信道一，二，三均周期占用，周期不同
        for i in range(self.channel_num):
            tmp = collections.deque(maxlen=self.time_step)
            for j in range(self.time_step):
                obs = data_in["channel_"+str(i+1)][j]
                tmp.append(obs)
            self.state.append(tmp)

        return np.array(self.state)

        # 情景二：假设信道一，二，三占用状态均符合特定分布随机变量

        # 情景三：混合占用模式

    def step(self, action:int, steps:int):  # 在steps下实行的动作，及steps值

        # 单智能体情况下：
        for i in range(self.channel_num):
            self.state[i].append(data_in["channel_"+str(i+1)][steps + self.time_step])

        if action == 0:  # 次级用户不进行信息传输
            reward = -0.1
            return np.array(self.state), reward, 0

        if data_in["channel_"+str(action)][steps + self.time_step]:
            reward = 2
        else: reward = -1
        return np.array(self.state), reward, 0




# dsa = DSA()
# state = dsa.reset()
# print(state)
# for i in range(7):
#     action = random.randint(0, 1)
#     state,reward,done,info = dsa.step(action)
#     print(state)
#
# a = [[1, 1, 1, 0, 1, 1, 0],[0,0,0,1,0,0,1]]
#
# plt.figure()
# sns.heatmap(a,cmap='Reds',yticklabels=['states', 'action'])
# plt.title('Environment_setting')
# plt.show()

# print(data_in)
# print(np.shape(data_in))
# print(data_in["channel_"+str(2)][5])
# dsa = DSA()
# state = dsa.reset()
# print(np.shape(state))
# for i in range(9):
#     action = random.randint(0, 3)
#     state, reward = dsa.step(action, i)
#     print(action,reward,str(i + dsa.time_step))
#     # print(state)



