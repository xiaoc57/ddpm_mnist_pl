import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import CosineAnnealingLR
from lightning_model.unet import UNet
from einops import rearrange, repeat
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings("ignore")

torch.set_float32_matmul_precision('high')

def extract(x_t, t, sh):
    
    # https://github.com/w86763777/pytorch-ddpm/blob/master/diffusion.py#L6
    v = "n" + " 1" * (len(sh) - 1)
    ans = rearrange(torch.gather(x_t, index = t, dim = 0), "n -> " + v).float()
    return ans

def to_img(x):
    img = np.clip(x, -1, 1)
    img_pred = (img + 1) / 2
    # img_pred = img_pred[0]
    img_pred = (img_pred * 255).astype(np.uint8)
    return img_pred

class DDPMSystem(L.LightningModule):
    
    
    def __init__(self, beta_1, beta_T, T, learning_rate = 1e-3, num_epoch = 50000):
        super().__init__()
        self.save_hyperparameters()

        # hard code
        ch=128
        ch_mult=[1, 2, 2, 2]
        attn=[1]
        num_res_blocks=2
        dropout=0.1
        
        self.register_buffer("betas", torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1. - alphas_bar))
        self.register_buffer("sqrt_recip_alphas_bar", torch.sqrt(1. / alphas_bar))
        
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        self.register_buffer('posterior_log_var_clipped', torch.log(torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer('posterior_mean_coef1', torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer('posterior_mean_coef2', torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

        
        self.model = UNet(
            T=self.hparams.T,
            ch=ch, 
            ch_mult=ch_mult, 
            attn=attn, 
            num_res_blocks=num_res_blocks,
            dropout=dropout
        )
        
        self.loss_fn = nn.MSELoss()
    
    def forward(self, x_0):
        t = torch.randint(self.hparams.T, size=(x_0.shape[0], ), device=self.device)
        noise = torch.randn_like(x_0, device=self.device)

        s1 = extract(self.sqrt_alphas_bar.to(self.device), t, x_0.shape)
        s2 = extract(self.sqrt_one_minus_alphas_bar.to(self.device), t, x_0.shape)

        x_t = (s1 * x_0 + s2 * noise)
        esp = noise
        esp_pred = self.model(x_t, t)
        return esp, esp_pred

    def training_step(self, batch, batch_idx):
        
        img = batch[0]
        img = F.pad(img, pad=(2, 2, 2, 2), mode='constant', value=0)

        esp, esp_pred = self(img)
        loss = self.loss_fn(esp, esp_pred)
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_lr", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])
        lrscheduler = CosineAnnealingLR(optimizer, self.hparams["num_epoch"], self.hparams["learning_rate"]/30)
        return [optimizer], [lrscheduler]
    
    def image_infer(self, batch, count = None, pred_type = "epsilon"):
        
        x_t = batch[0].to(self.device)
        x_t = F.pad(x_t, pad=(2, 2, 2, 2), mode='constant', value=0)
        x_t = torch.randn_like(x_t, device = self.device)
        
        pred_type_list = ["epsilon"]
        assert pred_type in pred_type_list
        
        b = x_t.shape[0]
        
        img_list = []

        with torch.no_grad():
            countt = 0
            for time_step in reversed(tqdm(range(self.hparams.T))):
                # if use tqdm, the reversed is blocked.
                time_step = self.hparams.T - time_step - 1  # Notice.
                
                countt += 1
                if count is not None and countt > count:
                    break
                t = torch.ones(b, device = self.device).long() * time_step
                eps_pred = self.model(x_t, t)
                
                x_0 = extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * (x_t - extract(self.sqrt_one_minus_alphas_bar, t, x_t.shape) * eps_pred)
                
                log_sigma = extract(self.posterior_log_var_clipped, t, x_t.shape)
                mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
                
                if time_step > 0:
                    noise = torch.randn_like(x_t)
                else:
                    noise = 0
                x_t = mean + torch.exp(0.5 * log_sigma) * noise

                if time_step % 10 == 0:
                    k = x_t[0][0].detach().cpu().numpy()
                    img_list.append(to_img(k))
                
            x_0 = x_t
            return torch.clip(x_0, -1, 1), img_list