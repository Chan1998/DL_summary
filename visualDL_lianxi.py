from visualdl import LogWriter
import numpy as np
from PIL import Image
import DSA_env as ENV
import matplotlib.pyplot as pyplot
import random

def random_crop(img):
    """获取图片的随机 100x100 分片
    """
    img = Image.open(img)
    w, h = img.size
    random_w = np.random.randint(0, w - 100)
    random_h = np.random.randint(0, h - 100)
    r = img.crop((random_w, random_h, random_w + 100, random_h + 100))
    return np.asarray(r)


if __name__ == '__main__':
    # 初始化一个记录器
    # with LogWriter(logdir="./log/image_test/train") as writer:
    #     for step in range(6):
    #         # 添加一个图片数据
    #         writer.add_image(tag="eye",
    #                          img=random_crop("./log/image_test/train/img.jpg"),
    #                          step=step)


    # value = [i/1000.0 for i in range(1000)]
    # # 初始化一个记录器
    # with LogWriter(logdir="./log/scalar_test/train") as writer:
    #     for step in range(1000):
    #         # 向记录器添加一个tag为`acc`的数据
    #         writer.add_scalar(tag="acc", step=step, value=value[step])
    #         # 向记录器添加一个tag为`loss`的数据
    #         writer.add_scalar(tag="loss", step=step, value=1/(value[step] + 1))




    # dsa = ENV.DSA()
    # with LogWriter(logdir="./log/image_test/train") as writer:
    #     state = dsa.reset()
    #     print(np.shape(state))
    #     for i in range(1000):
    #         action = random.randint(0, 3)
    #         state, reward, _ = dsa.step(action, i)
    #         # print(action,reward,str(i + dsa.time_step))
    #         # state = Image.fromarray(state.astype(np.uint8) * 255)
    #         # print(np.array(state))
    #         writer.add_image(tag="state",
    #                          img=state.astype(np.uint8) * 255,
    #                          step=i,
    #                          dataformats="HW")

    # img = Image.open("./log/image_test/train/img.jpg")
    # print(np.shape(img))
    # w, h = img.size
    # print(w,h)
    # # print(np.array(img)[100][100])
    # img.show()

    dsa = ENV.DSA()
    state = dsa.reset()
    pyplot.imshow(state)
    pyplot.show()
    state = Image.fromarray(state.astype(np.uint8) * 255)


    #
    print(np.array(state))
    state.show()




