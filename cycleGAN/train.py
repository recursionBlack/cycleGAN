import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch
from models import Generator, Discriminator
from utils import ReplayBuffer, LambdaLR, weight_init_normal
from datasets import ImageDataset
import itertools
import tensorboardX

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batchsize = 1
size = 256
lr = 0.0002
n_epoch = 200
epoch = 0
decay_epoch = 100

# networks
netG_A2B = Generator().to(device)
netG_B2A = Generator().to(device)
netD_A = Discriminator().to(device)
netD_B = Discriminator().to(device)

# loss
loss_GAN = torch.nn.MSELoss()
loss_cycle = torch.nn.L1Loss()
# 相似性的loss，主要用来判断生成器和真实数据的相似程度
loss_identity = torch.nn.L1Loss()

# optimizer  & LR
# 将两个生成器网络的优化器参数进行连接
opt_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                         lr=lr, betas=(0.5, 0.9999))
opt_DA = torch.optim.Adam(netD_A.parameters(), lr=lr, betas=(0.5, 0.9999))
opt_DB = torch.optim.Adam(netD_B.parameters(), lr=lr, betas=(0.5, 0.9999))

# 学习率衰减的方法
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(opt_G,
                                                   lr_lambda=LambdaLR(n_epoch,epoch, decay_epoch).step)
lr_scheduler_DA = torch.optim.lr_scheduler.LambdaLR(opt_DA,
                                                   lr_lambda=LambdaLR(n_epoch,epoch, decay_epoch).step)
lr_scheduler_DB = torch.optim.lr_scheduler.LambdaLR(opt_DB,
                                                   lr_lambda=LambdaLR(n_epoch,epoch, decay_epoch).step)


# 训练数据的路径
data_root = "datasets/apple2orange"
# 定义输入数据
input_A = torch.ones([1, 3, size, size],
                     dtype=torch.float).to(device)
input_B = torch.ones([1, 3, size, size],
                     dtype=torch.float).to(device)

# 定义label
label_real = torch.ones([1], dtype=torch.float,
                        requires_grad=False).to(device)
label_fake = torch.zeros([1], dtype=torch.float,
                        requires_grad=False).to(device)

# 定义好两个buffer
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# 定义log存放的路径
log_path = "logs"
# 定义log写手，不过会降低学习速率
# writer_log = tensorboardX.SummaryWriter(log_path)

transforms_ = [
    # 尺寸放大
    transforms.Resize(int(256 * 1.12), Image.BICUBIC),
    # 裁剪，需要先放大
    transforms.RandomCrop(256),
    # 随机翻转
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]

# num_worker数据读取的线程数
dataloader = DataLoader(ImageDataset(data_root, transforms_),
                        batch_size=batchsize,
                        shuffle=True,
                        num_workers=8)

# 计数器，用来打印log用到的
step = 0
for epoch in range(n_epoch):
    for i, batch in enumerate(dataloader):
        real_A = torch.tensor(input_A.copy_(batch['A']),
                              dtype=torch.float).to(device)
        real_B = torch.tensor(input_B.copy_(batch['B']),
                              dtype=torch.float).to(device)
        # 定义生成器
        opt_G.zero_grad()

        # 用真实的B，来生成一个B，并看生成的与真实的B的偏差
        same_B = netG_A2B(real_B)
        # loss越小越好
        loss_identity_B = loss_identity(same_B, real_B) * 5.0
        # A同理
        same_A = netG_B2A(real_A)
        loss_identity_A = loss_identity(same_A, real_A) * 5.0

        # 再用A生成假的B
        fake_B = netG_A2B(real_A)
        # 对于假的A，也要预测一个结果
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = loss_GAN(pred_fake, label_real)
        # 对A同理
        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = loss_GAN(pred_fake, label_real)

        # cycle loss
        # 需要保证cycle的一致性，利用生成的假的A和生成的假的B，
        # 来利用两个生成器，再分别恢复出来A和B，
        # 保证恢复出来的结果和原始的结果是一致的
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = loss_cycle(recovered_A, real_A) * 10.0
        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = loss_cycle(recovered_B, real_B) * 10.0

        # 最后拿到生成器整体的loss
        loss_G = loss_identity_A + loss_identity_B + \
                 loss_GAN_A2B + loss_GAN_B2A + \
                 loss_cycle_ABA + loss_cycle_BAB

        # 在定义好生成器后，对生成器进行反向操作, 对生成器参数进行优化
        loss_G.backward()
        opt_G.step()

        ##############################################
        # 定义判别器A
        opt_DA.zero_grad()

        # 使用判别器A，对真实的A进行预测
        pred_real = netD_A(real_A)
        # 预测的label应该和我们真实的label是一致的
        loss_D_real = loss_GAN(pred_real, label_real)

        # 定义生成的欺骗的数据的label
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        # 调用生成器A，对于生成数据的判别器的预测结果
        pred_fake = netD_A(fake_A.detach())
        # 对于判别器而言，我需要将它识别为一个负样本
        loss_D_fake = loss_GAN(pred_real, label_fake)

        # total loss
        # 判别器A的total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()
        opt_DA.step()

        ### 对B判别器进行相同的操作
        opt_DB.zero_grad()
        pred_real = netD_B(real_B)
        loss_D_real = loss_GAN(pred_real, label_real)

        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = loss_GAN(pred_real, label_fake)

        # total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()
        opt_DB.step()

        print("loss_G:{}, loss_G_identity:{}, loss_G_GAN:{},"
              "loss_G_cycle:{}, loss_D_A:{}, loss_D_B:{}".format(
            loss_G, loss_identity_A + loss_identity_B,
            loss_GAN_A2B + loss_GAN_B2A,
            loss_cycle_ABA + loss_cycle_BAB,
            loss_D_A, loss_D_B
        ))

        # writer_log.add_scalar("loss_G", loss_G, global_step=step + 1)
        # writer_log.add_scalar("loss_G_identity", loss_identity_A + loss_identity_B, global_step=step + 1)
        # writer_log.add_scalar("loss_G_GAN", loss_GAN_A2B + loss_GAN_B2A, global_step=step + 1)
        # writer_log.add_scalar("loss_G_cycle", loss_cycle_ABA + loss_cycle_BAB, global_step=step + 1)
        # writer_log.add_scalar("loss_D_A", loss_D_A, global_step=step + 1)
        # writer_log.add_scalar("loss_D_B", loss_D_B, global_step=step + 1)
        step += 1

    lr_scheduler_G.step()
    lr_scheduler_DA.step()
    lr_scheduler_DB.step()

    # 每训练一个epoch后保存一次模型
    torch.save(netG_A2B.state_dict(), "models/netG_A2B.path")
    torch.save(netG_B2A.state_dict(), "models/netG_B2A.path")
    torch.save(netD_A.state_dict(), "models/netD_A.path")
    torch.save(netD_B.state_dict(), "models/netD_B.path")