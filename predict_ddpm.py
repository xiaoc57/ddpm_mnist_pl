import os

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import datasets, transforms

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

import warnings
warnings.filterwarnings("ignore")

torch.set_float32_matmul_precision('high')

from lightning_model import DDPMSystem

import matplotlib.pyplot as plt

def prepare_data():
    # transforms for images
    transform=transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.1307,), (0.3081,))])
        
    # prepare transforms standard to MNIST
    mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
    mnist_train = [mnist_train[i] for i in range(2200)]
    
    mnist_train, mnist_val = random_split(mnist_train, [2000, 200])

    mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)
    mnist_test = [mnist_test[i] for i in range(3000,4000)]

    return mnist_train, mnist_val, mnist_test

if __name__ == "__main__":
    
    
    # 准备数据
    train, val, test = prepare_data()
    train_loader, val_loader, test_loader = DataLoader(train, batch_size=256), DataLoader(val, batch_size=64), DataLoader(test, batch_size=64)
    
    
    device = "cuda"
    
    # 加载模型
    model = DDPMSystem.load_from_checkpoint(checkpoint_path="D:\document\RayDiffusion\checkpoints\mnist\mnistddpm\epoch=4999.ckpt",
                                            strict=False)
    
    # 切换模型到评估模式
    model.eval()
    model.to(device)  # 根据你的环境选择适当的设备
    
    # test_loader = test_loader.to(device)
    # 进行推理
    # 禁用梯度计算
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            count = 1100
            batch_images = model.image_infer(batch, count)
            
            img = batch[0][0] # 1, 28, 28
            img = F.pad(img, pad=(2, 2, 2, 2), mode='constant', value=0)
            
            img_pred = (batch_images + 1) / 2
            img_pred = img_pred[0]
            
            break
    
    # 设置画布大小
    plt.figure(figsize=(10, 5))

    # 绘制第一张图像
    plt.subplot(1, 2, 1)  # 1行2列的图表中的第1个
    plt.imshow(img[0].cpu().numpy(), cmap='gray')  # squeeze 用于去除单维度条目，cmap='gray' 指定灰度色图
    plt.title('Image 1')
    plt.axis('off')  # 关闭坐标轴

    # 绘制第二张图像
    plt.subplot(1, 2, 2)  # 1行2列的图表中的第2个
    plt.imshow(img_pred[0].cpu().numpy(), cmap='gray')
    plt.title('Image 2')
    plt.axis('off')

    # 显示图像
    plt.show()
            
            
            