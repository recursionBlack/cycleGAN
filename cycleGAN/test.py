import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torch
from models import Generator, Discriminator
from datasets import ImageDataset
import os

from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# eval只需要定义两个生成器就可以
netG_A2B = Generator().to(device)
netG_B2A = Generator().to(device)

# 从训练好的文件里加载训练好的参数
netG_A2B.load_state_dict(torch.load("models/netG_A2B.path"))
netG_B2A.load_state_dict(torch.load("models/netG_B2A.path"))

netG_A2B.eval()
netG_B2A.eval()
size = 256

# 定义输入数据
input_A = torch.ones([1, 3, size, size],
                     dtype=torch.float).to(device)
input_B = torch.ones([1, 3, size, size],
                     dtype=torch.float).to(device)

transforms_ = [
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]

data_root = "datasets/apple2orange"
dataloader = DataLoader(ImageDataset(data_root, transforms_, "test"),
                        batch_size=1,shuffle=False,num_workers=8)

if not os.path.exists("outputs/A"):
    os.makedirs("outputs/A")
if not os.path.exists("outputs/B"):
    os.makedirs("outputs/B")

for i, batch in enumerate(dataloader):
    real_A = torch.tensor(input_A.copy_(batch['A']),
                          dtype=torch.float).to(device)
    real_B = torch.tensor(input_B.copy_(batch['B']),
                          dtype=torch.float).to(device)

    # 利用A生成B
    fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)
    # 利用B生成A
    fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)

    # 保存生成的图片
    save_image(fake_A, "outputs/A/{}.png".format(i))
    save_image(fake_B, "outputs/B/{}.png".format(i))
    print(i)
