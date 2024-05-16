import os

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms

from lightning_model import DDPMSystem

import matplotlib.pyplot as plt
import imageio
import warnings
warnings.filterwarnings("ignore")

torch.set_float32_matmul_precision('high')

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
    
    train, val, test = prepare_data()
    train_loader, val_loader, test_loader = DataLoader(train, batch_size=256), DataLoader(val, batch_size=64), DataLoader(test, batch_size=64)
    
    device = "cuda"

    model = DDPMSystem.load_from_checkpoint(
        checkpoint_path="./checkpoints/mnist/mnistddpm/epoch=2499.ckpt",strict=False)
    
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            batch_images, img_list = model.image_infer(batch, None)
            break
    imageio.mimwrite("result.gif", img_list)

            
            
            