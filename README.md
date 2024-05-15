# ddpm训练基于pytorchlightning
首先，我觉得这个框架用用就会了，其次真的省去了在多卡情况下训练的麻烦。
但是这个框架也有他的问题，我觉得有些时候并不好用，所以是否要选择学习这个框架需要斟酌

以下流程皆在windows11上，并安装了vs2022
vs2022非常建议开发者首先安装，这个repo应该是不需要的，但如果做一些比较变态的库安装例如需要编译的一些库的时候，就需要vs2022中的工具链了
cuda 也是必须的，建议安装cuda 11.8，本repo基于11.8

# first
use conda to create a env

    conda create -n mnist_ddpm python=3.9
    conda activate mnist_ddpm
# 2
安装 pytorch
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3 安装其他依赖
    pip install -r requirements.txt