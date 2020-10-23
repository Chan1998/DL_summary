import random
import pandas as pd
import numpy as np


# 场景一：#情景一：假设信道一，二，三均周期占用，周期不同
#生成excel
def state_generate(channel_num, time_slots, channel_period):
    state_data = np.zeros(shape=[time_slots, channel_num], dtype=int)
    for i in range(channel_num):
        period = channel_period[i]
        for j in range(time_slots):
            if j % period == 1:
                state_data[j,i] = 1 # 1表示主用户未占用，次用户可以进行频谱接入
    return state_data


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

