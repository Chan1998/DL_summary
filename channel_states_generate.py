import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 场景一：#情景一：假设信道一，二，三均周期占用，周期不同
#生成excel
# def state_generate(channel_num, time_slots, channel_period):
#     state_data = np.zeros(shape=[time_slots, channel_num], dtype=int)
#     for i in range(channel_num):
#         period = channel_period[i]
#         for j in range(time_slots):
#             if j % period == 1:
#                 state_data[j,i] = 1 # 1表示主用户未占用，次用户可以进行频谱接入
#     return state_data


#  # 第一种情况：周期【2.3.5】
#
# state_data = state_generate(3,10000,[2,3,5])
# print(np.shape(state_data))
#
#
# state_data = pd.DataFrame(state_data, columns=['channel_1', 'channel_2', 'channel_3'])
#
# # print(state_data)
#
# state_data.to_csv("./period_state235.csv", index=False)

# #第二种情况：周期【7，11，13】
#
# state_data = state_generate(3,10000,[7,11,13])
# print(np.shape(state_data))
#
#
# state_data = pd.DataFrame(state_data, columns=['channel_1', 'channel_2', 'channel_3'])
#
# # print(state_data)
#
# state_data.to_csv("./period_state71113.csv", index=False)

#第三种情况：周期【7，11，13】

# state_data = state_generate(8,10000,[8,9,10,11,12,13,14,15])
# print(np.shape(state_data))
#
#
# state_data = pd.DataFrame(state_data, columns=['channel_1', 'channel_2', 'channel_3', 'channel_4',
#                                                'channel_5', 'channel_6', 'channel_7', 'channel_8'])
#
# # print(state_data)
#
# state_data.to_csv("./period_state8.csv", index=False)

# 场景二：假设：
#               信道一，四周期性可用，周期为（【5，7】）
#               信道二仅在信道一下一时刻可用（强相关），信道三有30%几率与一信道相同，其他情况相反(负相关)
#               信道五与信道四在下一时刻80%相同（强正相关），信道六与信道四完全相反
#               信道七服从泊松分布（k=0:P=e**-lamda: k=1:p=lamda * e**-lamda,两者加和为一 ,），算出来lamda=0，假设空闲概率为30%
#               信道八完全随机
# 生成状态序列
def state_generate(channel_num=8, time_slots=11000):
    state_data = np.zeros(shape=[time_slots, channel_num], dtype=int)
    # 信道一，四：周期5，7
    for j in range(time_slots):
        if j % 5 == 1:
            state_data[j,0] = 1 # 1表示主用户未占用，次用户可以进行频谱接入
    for j in range(time_slots-3):
        if j % 7 == 1:
            state_data[j+3,3] = 1 # 1表示主用户未占用，次用户可以进行频谱接入

    # 信道二信道二仅在信道一下一时刻可用（强相关）：
    for j in range(time_slots-1):
        if state_data[j,0]:
            state_data[j+1,1] = 1 # 1表示主用户未占用，次用户可以进行频谱接入

    # 信道三有30 % 几率与一信道相同，其他情况相反(负相关)
    for j in range(time_slots-1):
        tmp = random.random()
        if tmp <= 0.3 and not state_data[j,0]:
            state_data[j+1,2] = 1 # 1表示主用户未占用，次用户可以进行频谱接入

    # 信道七服从泊松分布（k = 0:P = e ** -lamda: k = 1:p = lamda * e ** -lamda, 两者加和为一,），
    # 算出来lamda = 0，假设空闲概率为30 %
    #               信道八成功概率20%
    for j in range(time_slots):
        tmp = random.random()
        if tmp <= 0.3:
            state_data[j, 6] = 1
        if tmp >= 0.8:
            state_data[j, 7] = 1

    # 信道五与信道四在下一时刻80 % 相同（强正相关），信道六与信道四完全相反,且与七,八信道不共存
    for j in range(time_slots-1):
        tmp = random.random()
        if tmp >= 0.2 and state_data[j,3]:
            state_data[j+1,4] = 1 # 1表示主用户未占用，次用户可以进行频谱接入
        if not state_data[j,3] and not state_data[j,6] and not state_data[j,7]:
            state_data[j,5] = 1

    return state_data



# # 场景二，生成excel:
# state_data = state_generate()
# print(np.shape(state_data))
#
# state_data = pd.DataFrame(state_data, columns=['channel_1', 'channel_2', 'channel_3', 'channel_4',
#                                                'channel_5', 'channel_6', 'channel_7', 'channel_8'])
#
# # print(state_data)
#
# state_data.to_csv("./channel_data/correlated8.csv", index=False)

# 场景二：假设：
#               信道一，四周期性可用，周期为（【5，7】）
#               信道二仅在信道一下一时刻可用（强相关），信道三有30%几率与一信道相同，其他情况相反(负相关)
#               信道五与信道四在下一时刻80%相同（强正相关），信道六与信道四完全相反
#               信道七服从泊松分布（k=0:P=e**-lamda: k=1:p=lamda * e**-lamda,两者加和为一 ,），算出来lamda=0，假设空闲概率为30%
#               信道八完全随机
# 生成状态序列
def state_generate(channel_num=16, time_slots=11000):
    state_data = np.zeros(shape=[time_slots, channel_num], dtype=int)
    # 信道一，四：周期5，7
    for j in range(time_slots):
        if j % 5 == 1:
            state_data[j,0] = 1 # 1表示主用户未占用，次用户可以进行频谱接入
    for j in range(time_slots-3):
        if j % 7 == 1:
            state_data[j+3,3] = 1 # 1表示主用户未占用，次用户可以进行频谱接入

    # 信道二信道二仅在信道一下一时刻可用（强相关）：
    for j in range(time_slots-1):
        if state_data[j,0]:
            state_data[j+1,1] = 1 # 1表示主用户未占用，次用户可以进行频谱接入

    # 信道三有30 % 几率与一信道相同，其他情况相反(负相关)
    for j in range(time_slots-1):
        tmp = random.random()
        if tmp <= 0.3 and not state_data[j,0]:
            state_data[j+1,2] = 1 # 1表示主用户未占用，次用户可以进行频谱接入

    # 信道七服从泊松分布（k = 0:P = e ** -lamda: k = 1:p = lamda * e ** -lamda, 两者加和为一,），
    # 算出来lamda = 0，假设空闲概率为30 %
    #               信道八成功概率20%
    for j in range(time_slots):
        tmp = random.random()
        if tmp <= 0.3:
            state_data[j, 6] = 1
        if tmp >= 0.8:
            state_data[j, 7] = 1

    # 信道五与信道四在下一时刻80 % 相同（强正相关），信道六与信道四完全相反,且与七,八信道不共存
    for j in range(time_slots-1):
        tmp = random.random()
        if tmp >= 0.2 and state_data[j,3]:
            state_data[j+1,4] = 1 # 1表示主用户未占用，次用户可以进行频谱接入
        if not state_data[j,3] and not state_data[j,6] and not state_data[j,7]:
            state_data[j,5] = 1

    # 信道九，十二：周期9，7
    for j in range(time_slots):
        if j % 9 == 1:
            state_data[j, 8] = 1  # 1表示主用户未占用，次用户可以进行频谱接入
    for j in range(time_slots - 2):
        if j % 6 == 1:
            state_data[j + 2, 1] = 1  # 1表示主用户未占用，次用户可以进行频谱接入

        # 信道10：信道10仅在信道9下一时刻可用（强相关）：
    for j in range(time_slots - 1):
        if state_data[j, 8]:
            state_data[j + 1, 9] = 1  # 1表示主用户未占用，次用户可以进行频谱接入

    # 信道11有40 % 几率与9信道相同，其他情况相反(负相关)
    for j in range(time_slots - 1):
        tmp = random.random()
        if tmp <= 0.4 and not state_data[j, 8]:
            state_data[j + 1, 10] = 1  # 1表示主用户未占用，次用户可以进行频谱接入

        # 信道15服从泊松分布（k = 0:P = e ** -lamda: k = 1:p = lamda * e ** -lamda, 两者加和为一,），
    # 算出来lamda = 0，假设空闲概率为30 %
    #               信道16成功概率20%
    for j in range(time_slots):
        tmp = random.random()
        if tmp <= 0.3:
            state_data[j, 14] = 1
        if tmp >= 0.8:
            state_data[j, 15] = 1

        # 信道13与信道12在下一时刻80 % 相同（强正相关），信道14与信道12完全相反,且与15,16信道不共存
    for j in range(time_slots - 1):
        tmp = random.random()
        if tmp >= 0.2 and state_data[j, 11]:
            state_data[j + 1, 12] = 1  # 1表示主用户未占用，次用户可以进行频谱接入
        if not state_data[j, 11] and not state_data[j, 14] and not state_data[j, 15]:
            state_data[j, 13] = 1

    return state_data

 # 场景二，生成excel:
state_data = state_generate()
print(np.shape(state_data))

state_data = pd.DataFrame(state_data, columns=['channel_1', 'channel_2', 'channel_3', 'channel_4',
                                               'channel_5', 'channel_6', 'channel_7', 'channel_8',
                                               'channel_9', 'channel_10', 'channel_11', 'channel_12',
                                               'channel_13', 'channel_14', 'channel_15', 'channel_16'
                                               ])

# print(state_data)

state_data.to_csv("./channel_data/correlated16.csv", index=False)