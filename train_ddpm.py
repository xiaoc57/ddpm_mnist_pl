import os

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from lightning_model import DDPMSystem

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
    
    dataset_name = "mnist"
    exp_name = "mnistddpm"
    
    learning_rate = 1e-4
    epoch = 5000
    beta_1 = 1e-4
    beta_T = 0.02
    T = 1000

    train, val, test = prepare_data()
    train_loader, val_loader, test_loader = DataLoader(train, batch_size=256), DataLoader(val, batch_size=64), DataLoader(test, batch_size=64)
    
    # callbacks
    ckpt_cb = ModelCheckpoint(dirpath=f'checkpoints/{dataset_name}/{exp_name}', filename='{epoch:d}', every_n_epochs=500)
    lr_cb = LearningRateMonitor(logging_interval='epoch')
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1), lr_cb]
    # logger
    logger = TensorBoardLogger(save_dir=f"logs/{dataset_name}", name=exp_name)
    
    trainer = L.Trainer(max_epochs=epoch, callbacks = callbacks, logger=logger, precision=16)
    
    model = DDPMSystem(beta_1 = beta_1, beta_T = beta_T, T = T, num_epoch = epoch, learning_rate=learning_rate)
    trainer.fit(model=model, train_dataloaders=train_loader)