# DDPM-MNIST-Pytorchlightning
A simple method to train a ddpm model.


## Installation

I use the code on Windows with cuda 11.8. By the way, I use an A4500 graphics card, which has 20G of graphics memory.

```bash
conda create -n mnist_ddpm python=3.9
conda activate mnist_ddpm
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

In any case, I recommend that you install the corresponding version of pytorch yourself rather than using any of the commands in any repos.

## Running the Code

I recommend that you read the code before using it and modify the parameters according to your actual situation, such as adjusting the batch size according to the graphics memory.

### Training

```bash
python train_ddpm.py
```

There are some hyperparameters to tune.

* learning_rate
* T=1000
* epoch
* ...

However, the task is so simple that little tuning of hyperparameters is required.

### Testing
```bash
python predict_ddpm.py
```
<div align=center> 

![image](https://github.com/xiaoc57/ddpm_mnist_pl/blob/master/assets/result.gif)

</div>


## References
1. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
2. [63、必看！概率扩散模型(DDPM)与分数扩散模型(SMLD)的联系与区别](https://www.bilibili.com/video/BV1QG4y1674Q/?p=1&spm_id_from=pageDriver)
3. [Github Unofficial PyTorch implementation Denoising Diffusion Probabilistic Models](https://github.com/w86763777/pytorch-ddpm)
4. [pytorchlightning reference: how to implement instant ngp using pl](https://github.com/kwea123/ngp_pl)

## Citations

```bibtex
@misc{ho2020denoising,
    title   = {Denoising Diffusion Probabilistic Models},
    author  = {Jonathan Ho and Ajay Jain and Pieter Abbeel},
    year    = {2020},
    eprint  = {2006.11239},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```



