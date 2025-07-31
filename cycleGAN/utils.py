import random
import torch
import numpy as np

# 功能函数，搭建模型时需要用到的简单的脚本

def tensor2image(tensor):
    image = 127.5 * (tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    return image.astype(np.uint8)

class ReplayBuffer():
    """
    进行模型训练时，在cycleGAN中需要取拿到一些生成的数据
    利用这些生成的数据作为判别器的输入
    为了保证训练的稳定性，在抽取生成数据时，并没有直接把生成器生成的数据拿过来去用
    而是将已经生成好的数据，放到一个队列里去
    以队列的形式作为判别器的输入
    """

    def __init__(self, max_size=50):
        assert(max_size > 0), "Empty buffer or trying to create a buffer"
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            # 如果当前数据的尺寸小于最大尺寸的,就全放进来
            # 否则，就随机的，以0.5的概率决定是否加进来
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)

# 定义一个学习率衰减的方程
class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch)), "Decay must start be "
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    # 根据输入的epoch和epoch的总数，来对学习率进行衰减
    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset + self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

# 参数初始化的方程
def weight_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.2)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)
