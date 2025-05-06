import argparse
import datetime
import json
from PIL import Image
import time, os
import numpy as np
import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F
from models.unet import DiffusionUNet
from models.dit import DiT
from pytorch_msssim import ssim
from hdrutils.utils import radiance_writer, range_compressor, calculate_psnr, calculate_ssim
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.optimize import get_optimizer
import wandb
from omegaconf import OmegaConf
import torchvision.transforms as transforms
from tqdm import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.lp_utils import Upsample, RCAB, ReduceChannel, SpatialTemporalTransformerBlock, bchw_to_blc, blc_to_bchw

def check_size(x):
    _, _, img_h, img_w = x.shape
    img_h_32 = int(32 * np.ceil(img_h / 32.0))
    img_w_32 = int(32 * np.ceil(img_w / 32.0))
    x = F.pad(x, (0, img_w_32 - img_w, 0, img_h_32 - img_h), 'reflect')
    return x, img_h, img_w   

def dict_to_namespace(d):
    """
    将字典递归转换为 argparse.Namespace。
    """
    if isinstance(d, dict):
        return argparse.Namespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    else:
        return d

def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def detach_tensors(tensors):
    result = []
    for t in tensors:
        if isinstance(t, tuple):
            # 如果是 tuple，对每个元素递归 detach
            new_t = tuple(ti.detach() if torch.is_tensor(ti) else ti for ti in t)
            result.append(new_t)
        elif torch.is_tensor(t):
            # 如果是 tensor，直接 detach
            result.append(t.detach())
        else:
            # 其他类型保持不变
            result.append(t)
    return result

# 衡量图像在空间上的梯度变化
class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        # 水平方向梯度
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        # 垂直方向梯度
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        """将模型参数拷贝到 shadow 字典中，并保持设备一致"""
        if isinstance(module, nn.DataParallel) or isinstance(module, nn.parallel.DistributedDataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                # 确保克隆参数在相同设备上
                self.shadow[name] = param.data.clone().to(param.device)

    def update(self, module):
        """用当前模型参数更新 shadow 参数"""
        if isinstance(module, nn.DataParallel) or isinstance(module, nn.parallel.DistributedDataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                # 如果影子参数不在当前设备上，先移动过去
                if self.shadow[name].device != param.device:
                    self.shadow[name] = self.shadow[name].to(param.device)
                # 更新影子参数
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        """将 shadow 参数加载回模型"""
        if isinstance(module, nn.DataParallel) or isinstance(module, nn.parallel.DistributedDataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                if self.shadow[name].device != param.device:
                    self.shadow[name] = self.shadow[name].to(param.device)
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        """创建一个副本并应用 EMA 参数"""
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        elif isinstance(module, nn.parallel.DistributedDataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.parallel.DistributedDataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())

        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

class Lap_Pyramid_Conv(nn.Module):
    r"""
    Args:
        num_high (int): Number of high-frequency components
    """
    def __init__(self, LP_config):
        super(Lap_Pyramid_Conv, self).__init__()

        self.num_high = LP_config.num_high
        self.channels = LP_config.in_chans
        self.embed_dim = LP_config.embed_dim
        self.kernel = nn.Parameter(self.gauss_kernel(channels=self.embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=LP_config.drop_rate)
        
        # 3 -> 16
        self.conv_x = nn.Conv2d(self.channels, self.embed_dim, 3, 1, 1)
        self.conv_f = nn.Conv2d(self.embed_dim, self.channels, 3, 1, 1)

        self.channel_attention = nn.ModuleList(
            [RCAB(n_feat=2*self.embed_dim if i <self.num_high else self.embed_dim, reduction=2)
                 for i in range(self.num_high + 1)]
        )
        self.up_sample = nn.ModuleList(
            [Upsample(2*self.embed_dim) for _ in range(self.num_high-1)]
        )
        self.reduce_channel = nn.ModuleList(
            [ReduceChannel(2*self.embed_dim) for _ in range(self.num_high)]
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 0.02)
            if m.bias is not None:
                m.bias.data.normal_(0.0, 0.02)

    def gauss_kernel(self, channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256. # normalize
        kernel = kernel.repeat(channels, 1, 1, 1) # size -> [channels, 1, 5, 5]
        return kernel

    # downsamples the image by rejecting even rows and colums
    # https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=pyrdown#void%20cvPyrDown(const%20CvArr*%20src,%20CvArr*%20dst,%20int%20filter)
    def downsample(self, x):
        return x[:, :, ::2, ::2] # downsamples the image by rejecting even rows and columns.

    def upsample(self, x):
        r"""it upsamples the source image by injecting even zero rows and columns and 
        then convolves the result with the same kernel as in downsample() multiplied by 4.
        https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=pyrdown#pyrup
        """
        # 以下面if __name__ == "__main__"的输入为例
        # -----------------------------------
        # inject even zero colums
        # x.shape=[1, 3, 132, 92] -> [1, 3, 132, 184]
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        # [1, 3, 132, 184] -> [1, 3, 264, 92]
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        # [1, 3, 92, 264]
        cc = cc.permute(0, 1, 3, 2)
        # ----------------------------------

        # ----------------------------------
        # inject even zero rows
        # cat([1, 3, 92, 264], [1, 3, 92, 264], dim=3) -> [1, 3, 92, 528]
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        # [1, 3, 184, 264]
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        # [1, 3, 264, 184]
        x_up = cc.permute(0, 1, 3, 2)
        # ----------------------------------

        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = F.pad(img, (2, 2, 2, 2), mode='reflect')
        # out = F.conv2d(img, kernel, groups=img.shape[1])
        # return out

        # https://www.pudn.com/news/6228cd129ddf223e1ad105c7.html
        kernel = kernel.to(img.device)
        return F.conv2d(img, kernel, groups=img.shape[1])

    def pyramid_decom(self, img):
        """
        High Low
        """
        current = self.conv_x(img)
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel) # Blurs an image with a gaussian kernel
            down = self.downsample(filtered) # downsample the blurred image, i.e., Gaussian Pyramid
            up = self.upsample(down) # Upsamples the downsampled image and then blurs it
            # --------------------
            # if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
            # --------------------
            if up.shape != current.shape:
                # -------------------
                # up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
                # -------------------
                up = F.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up # Laplacial Pyramid
            # high-freq
            # pyr.append((diff, current))
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr
    # pyramid_recons
    def pyramid_recons(self, pyr):
        image = pyr[-1] # pyr=[h_0^hat, h_1^hat, h_2^hat, I_3^hat]
        # print("***********************")
        # print(image.size())
        # print(len(pyr[:-1]))
        for i, level in enumerate(reversed(pyr[:-1])):
            # https://www.jianshu.com/p/e7de4cd92f68
            up = self.upsample(image)
            # if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
            if up.shape != level.shape:
                # up = nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
                up = F.interpolate(up, size=(level.shape[2], level.shape[3]))
            # if i == len(pyr[:-1]) - 2:
            #     image = self.conv_LL_LL(up) + level
            # elif i == len(pyr[:-1]) - 1:
            #     image = self.conv_LL(up) + level
            # else:
            image = up + level
        return image
    def forward(self, pyr, x):
        # B 3 H W
        # pyr = self.pyramid_decom(x_list)
        pyr_low = pyr[-1]
        ############
        pyr_low_up = F.interpolate(pyr_low, scale_factor=2, mode='nearest')
        pyr_result = []
        for i in range(self.num_high):
            pyr_high = pyr[self.num_high-(i+1)]
            assert pyr_high.shape == pyr_low_up.shape, print(pyr_high.shape, pyr_low_up.shape)
            # 2 * embed_dim
            pyr_high_with_low = torch.cat([pyr_high, pyr_low_up], dim=1)
            pyr_high_with_low_size = (pyr_high_with_low.shape[2:])
            # 2 * embed_dim
            pyr_high_with_low = torch.cat([pyr_high, pyr_low_up], dim=1)
            pyr_high_with_low_size = (pyr_high_with_low.shape[2:])
            pyr_high_with_low = bchw_to_blc(self.channel_attention[i](pyr_high_with_low))
            pyr_high_with_low = self.pos_drop(pyr_high_with_low)
            # layers
            result_highfreq = blc_to_bchw(pyr_high_with_low, pyr_high_with_low_size)
            # 
            if i < self.num_high-1:
                # pyr_low_up = F.interpolate(result_highfreq, size=pyr[self.num_high-(i+2)].shape[2:])
                # pyr_low_up = F.interpolate(result_highfreq, scale_factor=2, mode='nearest')
                pyr_low_up = self.up_sample[i](result_highfreq)
            # pre_high, pyr_low_up = self.up_sample[i](pyr_high_with_low)
            # pyr_result.append(pre_high)
            result_highfreq = self.reduce_channel[i](result_highfreq)
            setattr(self, f'result_highfreq_{str(i)}', result_highfreq)
        
        for i in reversed(range(self.num_high)):
            result_highfreq = getattr(self, f'result_highfreq_{str(i)}')
            pyr_result.append(result_highfreq)

        pyr_result.append(pyr_low)
        # self.pyramid_recons(pyr_result)
        r = self.pyramid_recons(pyr_result)
        return x + self.conv_f(r)


class SpatialAttentionModule(nn.Module):
    def __init__(self, dim):
        super(SpatialAttentionModule, self).__init__()
        self.att1 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, bias=True)
        self.att2 = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU()

    def forward(self, x1, x2):
        f_cat = torch.cat((x1, x2), 1)
        att_map = torch.sigmoid(self.att2(self.relu(self.att1(f_cat))))
        return att_map

class Net(nn.Module):
    def __init__(self, args, config):
        super(Net, self).__init__()

        self.args = args
        self.config = config
        self.vae = Lap_Pyramid_Conv(LP_config=self.config.VAE.LP)
        self.Unet = DiT(config)

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        # shape: torch.Size(num_diffusion_timesteps, )
        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = self.betas.shape[0]

    @staticmethod
    def compute_alpha(beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def sample_training(self, x_cond, b, eta=0.):
        # ddim
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        n, c, h, w = x_cond.shape
        seq_next = [-1] + list(seq[:-1])
        x = torch.randn(n, c, h, w, device=x_cond.device)
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            
            at = self.compute_alpha(b, t.long())
            at_next = self.compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)

            et = self.Unet(torch.cat([x_cond, xt], dim=1), t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to(x.device))

        return xs[-1]
    
    # 隐空间扩散
    def forward(self, x, gt):
        x_lp = self.vae.pyramid_decom(x)

        data_dict = {}
        ########################
        # input_lp = self.lp.pyramid_decom(x)
        input_high = x_lp[-2]
        ########################
        b = self.betas.to(input_high.device)

        # 对称时间步序列：结合了随机采样和对称补集的优点，能够覆盖正向和反向扩散的不同阶段
        t = torch.randint(low=0, high=self.num_timesteps, size=(input_high.shape[0] // 2 + 1,)).to(input_high.device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:input_high.shape[0]].to(input_high.device)
        a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)

        e = torch.randn_like(input_high)

        if self.training:
            # gt_lp= self.lp.pyramid_decom(label)
            # gt_high0, gt_LL = gt_lp[0]
            gt_lp = self.vae.pyramid_decom(gt)
            gt_high = gt_lp[-2]

            xet = gt_high * a.sqrt() + e * (1.0 - a).sqrt()
            noise_output = self.Unet(torch.cat([input_high, xet], dim=1), t.float())
            denoise_high = self.sample_training(input_high, b)

            # cat(input_LL_LL, denoise_high1)
            # up -> LL
            # cat(input_high0, LL)
            # conv3
            #
            # denoise_high1 = inverse_data_transform(denoise_high1)
            # pred_x, pred_LL = self.recon(input_lp, denoise_high1)
            # cl_pred_x = torch.clamp(pred_x, min=0, max=1)

            # data_dict["input_lp"] = input_lp               
            # data_dict["gt_lp"] = gt_lp

            x_cond_pred_list = x_lp[:-2] + [denoise_high, x_lp[-1]]
            pred_x = self.vae(x_cond_pred_list, x)
            data_dict["pred_x"] = pred_x
            data_dict["pred"] = denoise_high
            data_dict["noise_output"] = noise_output
            data_dict["e"] = e
            data_dict["gt"] = gt_high

        else:
            # label 随意填充为 zeros_like(input_img) 
            denoise_high = self.sample_training(input_high, b)
            x_cond_pred_list = x_lp[:-2] + [denoise_high, x_lp[-1]]
            pred_x = self.vae(x_cond_pred_list, x)
            # pred_x, _ = self.recon(input_lp, denoise_high1)
            # cl_pred_x = torch.clamp(pred_x, min=0, max=1)
            data_dict["pred"] = denoise_high
            data_dict["pred_x"] = pred_x

        return data_dict
    
class DDM(nn.Module):
    def __init__(self, configs, args, use_ddp=True, use_accelerate=True):
        super().__init__()
        self.configs = configs
        self.args = args
        self.model = Net(config=configs, args=args)
        if use_ddp:
            self.model = DDP(self.model.to(self.args.gpu), 
                            device_ids=[self.args.gpu],
                            find_unused_parameters=True)
        if use_accelerate:
            self.accelerator = Accelerator(
                mixed_precision=self.configs.accelerator.mixed_precision,
                gradient_accumulation_steps=self.configs.accelerator.gradient_accumulation_steps,
                log_with="wandb" if self.configs.wandb.is_use_wandb else None,
                kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
            )
            # self.vae = Lap_Pyramid_Conv(LP_config=self.configs.VAE.LP)
            self.args.gpu = self.accelerator.device
            if self.accelerator.use_distributed:
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
                # self.vae = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.vae)
            # self.optimizer_vae, self.scheduler_vae = get_optimizer(self.configs, self.vae.parameters())
            # self.ema_helper_vae = EMAHelper()
            # self.ema_helper_vae.register(self.vae) 

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)
        #
        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.TV_loss = TVLoss()
        self.optimizer, self.scheduler = get_optimizer(self.configs, self.model.parameters())

        #
        self.start_epoch, self.step = 0, 0
        # 
        self.model_sam = []
        self.log_file = None
    
    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = torch.load(load_path, map_location=None)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        if "ema_helper" in checkpoint:       
            self.ema_helper.load_state_dict(checkpoint['ema_helper'])
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint {}".format(load_path))
    
    @torch.no_grad()
    def restor(self, val_loader, device, resume=None):
        if resume is not None:
            self.load_ddm_ckpt(resume, ema=True)
        self.model.to(device)
        test_results = {}
        test_results['psnr_l'] = []
        test_results['ssim_l'] = []
        for data_dict in tqdm(val_loader, desc="Sampling validation patches", total=len(val_loader)):

            input0, img_h, img_w = check_size(data_dict[0][:, :3, :, :])
            label, _, _ = check_size(data_dict[0][:, 3:, :, :])
            
            x0 = input0.to(device)
            y = label.to(device)
            if len(x0.shape) == 3:
                x0 = x0.unsqueeze(0)
                y = y.unsqueeze(0)
                            
            with torch.no_grad():
                out = self.model(x0, y)

            pred_x_tensor = torch.clamp(out["pred_x"], min=0, max=1)
            # 1 3 1000 1500
            pred_x_tensor = pred_x_tensor[:, :, :img_h, :img_w]

            # 指标计算（逐样本计算）
            pred_imgs = out["pred_x"].detach().cpu().numpy().astype(np.float32)  # [B, C, H, W]
            labels = y.detach().cpu().numpy().astype(np.float32)  # [B, C, H, W]
            
            for b in range(pred_imgs.shape[0]):
                pred_img = pred_imgs[b]  # [C, H, W]
                label = labels[b]  # [C, H, W]
                # PSNR计算
                scene_psnr_l = calculate_psnr(label, pred_img, data_range=1.0)
                # SSIM计算
                pred_img_255 = np.clip(pred_img * 255.0, 0., 255.).transpose(1, 2, 0)
                label_255 = np.clip(label * 255.0, 0., 255.).transpose(1, 2, 0)
                scene_ssim_l = calculate_ssim(pred_img_255, label_255, data_range=255.0)
                
                test_results['psnr_l'].append(scene_psnr_l)
                test_results['ssim_l'].append(scene_ssim_l)
        
        print(f"Val, Datasets len{len(val_loader)}", \
            f"psnr-l: {np.mean(test_results['psnr_l']):.4f}", \
            f"ssim-l: {np.mean(test_results['ssim_l']):.4f}")

    def train(self, train_loader, val_loader, train_sampler, dict_configs):
        if self.args.gpu == 0:
            current_date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")  # 格式：YYYY-MM-DD
            date_dir = os.path.join(self.configs.data.ckpt_dir, 
                                self.configs.work_name + "_" + current_date)
            
            if not self.configs.DEBUG:
                os.makedirs(date_dir, exist_ok=True) 
            
            self.log_file = os.path.join(date_dir, "train_log.txt")
            self.save_log(self.log_file, dict_configs, indent=4)
            
            if self.configs.wandb.is_use_wandb:
                os.environ["WANDB_API_KEY"] = self.configs.wandb.APIkeys 
                # crop_match = re.search(r"crop\d+", self.configs.data.train.sub_set)
                wandb.init(project=self.configs.work_name, config=dict_configs, 
                            name=f"{self.args.world_size}GPUs" + \
                            f"_{self.configs.training.batch_size}Batch_" + current_date,
                            entity=self.configs.wandb.entity)
                            # dir=os.path.join(self.configs.wandb.dir_root, 
                            #     self.configs.work_name + "_" + current_date))
            print(
                f"""
                ╔{'═' * 60}╗
                ║ {'Training:':<15} {self.configs.work_name:<43}║
                ║ {'Model Name:':<15} {self.configs.model_name:<43}║
                ║ {'Data Name:':<15} {self.configs.data.name:<43}║
                ║ {'Data Type:':<15} {self.configs.data.type:<43}║
                ╚{'═' * 60}╝
                """
            )
            # 同步所有进程
            # torch.distributed.barrier()
        
        for epoch in range(self.start_epoch, self.configs.training.n_epochs):
            train_sampler.set_epoch(epoch)
            data_start = time.time()
            data_time = 0

            self.model.train()
            for data_dict in train_loader:
                x = data_dict[0].to(self.args.gpu)  
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x

                data_time += time.time() - data_start

                self.step += 1

                output_dict = self.model(x[:, :3, :, :], x[:, 3:, :, :])
                # noise_loss, photo_loss, frequency_loss = self.estimation_loss(output_dict['x'], 
                #                                                               output_dict)
                noise_loss, photo_loss, frequency_loss = self.estimation_loss_lp(output_dict)
                
                loss = noise_loss + photo_loss + frequency_loss                 
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()
                
                if self.step % self.configs.training.log_freq == 0 and self.args.gpu == 0\
                    and self.step != 0:
                        
                    noise_loss = self.sync_loss(noise_loss)
                    photo_loss = self.sync_loss(photo_loss)
                    frequency_loss = self.sync_loss(frequency_loss)
                    loss = noise_loss + photo_loss + frequency_loss    
                    
                    ttime = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                    log_entry ={
                        "date": ttime,
                        "epoch": epoch,
                        "step": self.step,
                        "lr": self.scheduler.get_last_lr()[0],
                        "noise_loss": noise_loss,
                        "photo_loss": photo_loss,
                        "frequency_loss": frequency_loss,
                        "loss": loss
                    }
                    self.save_log(self.log_file, log_entry)
                 
                    print("[{}] epoch:{}, step:{}, lr:{:.6f}, noise_loss:{:.4f}, photo_loss:{:.4f}, "
                          "frequency_loss:{:.4f}, loss:{:.4f}".format(ttime,
                                                        epoch, self.step, self.scheduler.get_last_lr()[0],
                                                         noise_loss, photo_loss,
                                                         frequency_loss, loss))
                    if self.configs.wandb.is_use_wandb:
                        wandb.log({"epoch": epoch, 
                                   "step": self.step, 
                                   "lr": self.scheduler.get_last_lr()[0],
                                   "noise_loss": noise_loss,
                                   "photo_loss": photo_loss,
                                   "frequency_loss": frequency_loss,
                                   "loss": loss})
                    # torch.distributed.barrier()
                    
                if self.step % self.configs.training.validation_freq == 0 and self.step != 0:
                    # 切换到评估模式
                    self.model.eval()

                    # 验证集采样（仅主进程）
                    if self.args.gpu == 0:
                        self.sample_validation_patches(val_loader, self.step, date_dir)
                        if not self.configs.DEBUG:
                            save_file_path = os.path.join(date_dir, "pth",'model_{}.pth'.format(self.step))
                            os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
                            torch.save({
                                #'step': self.step,
                                #'epoch': epoch + 1,
                                'state_dict': self.model.state_dict(),  # 保存 .module 的状态字典
                                #'optimizer': self.optimizer.state_dict(),
                                #'scheduler': self.scheduler.state_dict(),
                                #'ema_helper': self.ema_helper.state_dict(),
                                'param': self.args,
                                'config': self.configs
                            }, save_file_path)
                            print(f"Model saved successfully at step {self.step}")

                    # 同步所有进程
                    # torch.distributed.barrier()
                    # 切换回训练模式
                    self.model.train()

            self.scheduler.step()
        
        wandb.finish()
        self.find_best_model(date_dir)
    
    def train_accel(self, train_loader, val_loader, dict_configs):
        """
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=per_gpu_batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True
            )
        """
        if self.accelerator.is_main_process:
            current_date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")  # 格式：YYYY-MM-DD
            date_dir = os.path.join(self.configs.data.ckpt_dir, 
                                self.configs.work_name + "_" + current_date)
            
            if not self.configs.DEBUG:
                os.makedirs(date_dir, exist_ok=True) 
            
            self.log_file = os.path.join(date_dir, "train_log.txt")
            self.save_log(self.log_file, dict_configs, indent=4)
            if self.configs.wandb.is_use_wandb:
                os.environ["WANDB_API_KEY"] = self.configs.wandb.APIkeys
                self.accelerator.init_trackers(
                    project_name=self.configs.work_name,
                    config=dict_configs,
                    init_kwargs={"wandb": {"entity": self.configs.wandb.entity, \
                                        "name": f"{self.accelerator.num_processes}GPUs" + \
                                        f"_{self.configs.training.batch_size}Batch_" + current_date}}
                )
            print(
                f"""
                ╔{'═' * 60}╗
                ║ {'Training:':<15} {self.configs.work_name:<43}║
                ║ {'Model Name:':<15} {self.configs.model_name:<43}║
                ║ {'Data Name:':<15} {self.configs.data.name:<43}║
                ║ {'Data Type:':<15} {self.configs.data.type:<43}║
                ╚{'═' * 60}╝
                """
            )
        
        # self.model, self.vae, self.optimize, self.scheduler, self.optimizer_vae, self.scheduler_vae, train_loader, val_loader = \
        #     self.accelerator.prepare(self.model, self.vae, self.optimizer, self.scheduler, self.optimizer_vae, self.scheduler_vae, train_loader, val_loader)
        self.model, self.optimize, self.scheduler, train_loader, val_loader = \
            self.accelerator.prepare(self.model, self.optimizer, self.scheduler, train_loader, val_loader)

        for epoch in range(self.configs.training.n_epochs):
            self.model.train()
            for data_dict in tqdm(train_loader, disable=not self.accelerator.is_main_process, \
                    leave=False, desc=f"Epoch {epoch + 1}: ", total=len(train_loader)):
                x = data_dict[0]
                # x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                x_cond = x[:, :3, :, :]
                y = x[:, 3:, :, :]

                x_cond, imgh, imgw = check_size(x_cond)
                y, _, _ = check_size(y)

                self.step += 1
                # 隐空间压缩
                with self.accelerator.accumulate(self.model), self.accelerator.autocast():
                    # un_warp_vae = self.accelerator.unwrap_model(self.vae)
                    # x_cond_lp = un_warp_vae.pyramid_decom(x_cond)
                    # y_lp = un_warp_vae.pyramid_decom(y)

                    # input_high, gt_high = x_cond_lp[-2].detach(), y_lp[-2].detach()
                    output = self.model(x_cond, y)
                    noise_loss, pred_loss = self.l2_loss(output["noise_output"], output["e"]), \
                                            self.l1_loss(output["pred"], output["gt"])
                    loss_df = self.configs.loss.noise_loss_w * noise_loss + \
                                self.configs.loss.pred_loss_w * pred_loss
                                        
                    pred_x = output["pred_x"][:, :, :imgh, :imgw]
                    y = y[:, :, :imgh, :imgw]

                    photo_loss, ssim_loss = self.l1_loss(pred_x, y), (1 - ssim(pred_x, y, data_range=1.0))
                    loss_vae = self.configs.loss.photo_loss_w * photo_loss + \
                                self.configs.loss.ssim_loss_w * ssim_loss
                    
                    self.accelerator.backward(loss_df + loss_vae)
                    self.optimize.step()
                    self.optimize.zero_grad()

                    if self.step % self.configs.training.log_freq == 0 and self.accelerator.is_main_process and self.step!= 0:
                        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        tqdm.write(
                            f"[{current_time}] "
                            f"Epoch {epoch + 1}: step {self.step} | "
                            f"noise_loss: {noise_loss.item():.4f} | "
                            f"pred_z_loss: {pred_loss.item():.4f} | "
                            f"photo_loss: {photo_loss.item():.4f} | "
                            f"ssim_loss: {ssim_loss.item():.4f} | "
                            f"loss_diffusion: {loss_df.item():.4f} | "
                            f"loss_vae: {loss_vae.item():.4f}"
                        )

                        self.accelerator.log({
                            "noise_loss": noise_loss.item(),
                            "pred_z_loss": pred_loss.item(),
                            "photo_loss": photo_loss.item(),
                            "ssim_loss": ssim_loss.item(),
                            "loss_diffusion": loss_df.item(),
                            "loss_vae": loss_vae.item(),
                            "step": self.step,
                            "epoch": epoch
                        }, step=self.step)
                        
                        self.ema_helper.update(self.accelerator.unwrap_model(self.model))
                        # self.ema_helper_vae.update(self.accelerator.unwrap_model(self.vae))

                    if self.step % self.configs.training.validation_freq == 0 and self.step!= 0:
                        # 切换到评估模式
                        self.model.eval()
                        # self.vae.eval()
                        # 验证集采样（仅主进程）
                        if self.accelerator.is_main_process:
                            ssims, psnrs = [], []
                            val_pred_x = None
                            val_y = None
                            val_pred_z = None
                            for val_x in tqdm(val_loader, total=len(val_loader), desc="Validation", disable=not self.accelerator.is_local_main_process):
                                x = val_x[0]
                                with torch.no_grad():
                                    val_x_cond = x[:, :3, :, :]
                                    val_y = x[:, 3:, :, :]

                                    val_x_cond, imgh, imgw = check_size(val_x_cond)
                                    val_y, _, _ = check_size(val_y)

                                    val_output = self.model(val_x_cond, None)
                                    val_pred_z = val_output["pred"]
                                    val_pred_x = val_output["pred_x"][:, :, :imgh, :imgw]
                                    val_pred_x = torch.clamp(val_pred_x, min=0, max=1)
                                    
                                    # RGB
                                    val_pred_x = torch.squeeze(val_pred_x.detach()).cpu().numpy().transpose(1, 2, 0)
                                    val_y = torch.squeeze(val_y.detach()).cpu().numpy().transpose(1, 2, 0)

                                    ssim_ = compare_ssim(val_pred_x, val_y, multichannel=True, channel_axis=2, data_range=1.0)
                                    psnr_ = compare_psnr(val_pred_x, val_y)

                                    ssims.append(ssim_)
                                    psnrs.append(psnr_)
                            
                            self.accelerator.log({
                                "val_ssim": np.mean(ssims),
                                "val_psnr": np.mean(psnrs),
                                "step": self.step,
                                "epoch": epoch
                            }, step=self.step)

                            val_pred_x = (val_pred_x * 255).astype(np.uint8)
                            val_y = (val_y * 255).astype(np.uint8)
                            val_pred_z = torch.squeeze(val_pred_z.detach()).cpu().numpy().transpose(1, 2, 0)
                            val_pred_z = (val_pred_z * 255).astype(np.uint8)

                            self.accelerator.log({
                                "Deblur Image" : wandb.Image(Image.fromarray(val_pred_x, mode="RGB"), caption="val_pred_x"),
                                "Deblur Image (z)" : wandb.Image(Image.fromarray(val_pred_z, mode="RGB"), caption="val_pred_z")
                                #"GT Image" : wandb.Image(Image.fromarray(val_y, mode="RGB"), caption="val_y")
                            })

                            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            tqdm.write(
                                f"[{current_time}] "
                                f"Epoch {epoch + 1}| "
                                f"val_ssim: {np.mean(ssims):.4f} | "
                                f"val_psnr: {np.mean(psnrs):.4f}"
                            )

                        # 切换回训练模式
                        self.model.train()
                        # self.vae.train()


            self.scheduler.step()
            # self.scheduler_vae.step()     
    
    def estimation_loss(self, x, output):

        input_high0, input_high1, gt_high0, gt_high1 = output["input_high0"], output["input_high1"],\
                                                       output["gt_high0"], output["gt_high1"]

        pred_LL, gt_LL, pred_x, noise_output, e = output["pred_LL"], output["gt_LL"], output["pred_x"],\
                                                  output["noise_output"], output["e"]

        gt_img = x[:, 3:, :, :].to(self.args.gpu)
        # =============noise loss==================
        noise_loss = self.l2_loss(noise_output, e)

        # =============frequency loss==================
        frequency_loss = 0.1 * (self.l2_loss(input_high0, gt_high0) +
                                self.l2_loss(input_high1, gt_high1) +
                                self.l2_loss(pred_LL, gt_LL)) +\
                         0.01 * (self.TV_loss(input_high0) +
                                 self.TV_loss(input_high1) +
                                 self.TV_loss(pred_LL))

        # =============photo loss==================
        content_loss = self.l1_loss(pred_x, gt_img)
        ssim_loss = 1 - ssim(pred_x, gt_img, data_range=1.0)

        photo_loss = content_loss + ssim_loss

        return noise_loss, photo_loss, frequency_loss

    def estimation_loss_lp(self, output):

        input_lp = output["input_lp"]
        gt_lp = output["gt_lp"]
        noise_output = output["noise_output"]
        e = output["e"]
        pred_high1 = output["pred_high1"]
        pred_x = output["pred_x"]

        gt_img = output["gt"]
        # =============noise loss==================
        noise_loss = self.l2_loss(noise_output, e)

        # =============frequency loss==================
        input_high0, _ = input_lp[0]
        input_high1, _ = input_lp[1] 
        
        _, gt_LL = gt_lp[0]
        gt_high1, _ = gt_lp[1]
        
        # pred_LL = output["pred_LL"]
        # self.l2_loss(pred_LL, gt_LL)) 
        frequency_loss = 0.1 * (#self.l2_loss(input_high0, gt_high0) +
                                self.l2_loss(pred_high1, gt_high1)) +\
                         0.01 * (self.TV_loss(input_high0) +
                                 self.TV_loss(input_high1) +
                                 self.TV_loss(pred_high1))

        # =============photo loss==================
        content_loss = self.l1_loss(pred_x, gt_img)
        ssim_loss = 1 - ssim(pred_x, gt_img, data_range=1.0)

        photo_loss = content_loss + ssim_loss

        return self.configs.loss.noise_loss_w * noise_loss, \
            self.configs.loss.photo_loss_w * photo_loss, \
            self.configs.loss.frequency_loss_w * frequency_loss
    
    def sample_validation_patches(self, val_loader, step, date_dir):
        if not self.configs.DEBUG:
            image_folder = os.path.join(date_dir, "validation")
            os.makedirs(image_folder, exist_ok=True)
        print(f"Processing a single batch of validation images at step: {step}")
        
        test_results = {}
        test_results['psnr_l'] = []
        test_results['ssim_l'] = []
        # test_results['psnr_mu'] = []
        # test_results['ssim_mu'] = []
            
        save_pred_img = None
        
        # print(f"val len: {len(val_loader)}")
        
        for i, data_dict in tqdm(enumerate(val_loader), desc="Sampling validation patches", total=len(val_loader)):

            input0, img_h, img_w = check_size(data_dict[0][:, :3, :, :])
            label, _, _ = check_size(data_dict[0][:, 3:, :, :])
            
            x0 = input0.to(self.args.gpu)
            y = label.to(self.args.gpu)
            if len(x0.shape) == 3:
                x0 = x0.unsqueeze(0)
                y = y.unsqueeze(0)
                            
            with torch.no_grad():
                out = self.model(x0, y)

            pred_x_tensor = torch.clamp(out["pred_x"], min=0, max=1)
            # 1 3 1000 1500
            pred_x_tensor = pred_x_tensor[:, :, :img_h, :img_w]

            for b in range(pred_x_tensor.shape[0]):  # 遍历batch维度
                pred_x = torch.squeeze(pred_x_tensor[b].detach().cpu()).numpy().astype(np.float32)
                save_pred_img = (pred_x * 255).astype(np.uint8)
                save_pred_img = cv2.cvtColor(save_pred_img.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)

                # 保存/记录图像（每20个batch保存一次）
                # if not self.configs.DEBUG and i % 20 == 0:
                #     cv2.imwrite(os.path.join(image_folder, f"{step}_{i}_{b}.png"), save_pred_img)
                
                _y = torch.squeeze(y[b].detach().cpu()).numpy().astype(np.float32)
                save_pred_gt = (_y * 255).astype(np.uint8)
                save_pred_gt = cv2.cvtColor(save_pred_gt.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                
                # if self.configs.wandb.is_use_wandb and i % 20 == 0:
                #     wandb.log({
                #         f"HDR_MU_IMG_{i}_{b}": wandb.Image(save_pred_img, caption=f"val_{step}_{i}_{b}")
                #     })
                    # 如果需要标注，可以在图像上添加文字说明
                combined_image = cv2.hconcat([save_pred_img, save_pred_gt])
                combined_image = cv2.putText(
                    combined_image,
                    "Pred", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )
                combined_image = cv2.putText(
                    combined_image,
                    "GT", (save_pred_img.shape[1] + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )

                # 使用 wandb 记录合并后的图像
                if self.configs.wandb.is_use_wandb and i % 20 == 0:
                    wandb.log({
                        f"HDR_MU_IMG_GT_{i}_{b}": wandb.Image(combined_image, caption=f"val_{step}_{i}_{b}")
                    })
    
            # 指标计算（逐样本计算）
            pred_imgs = out["pred_x"].detach().cpu().numpy().astype(np.float32)  # [B, C, H, W]
            labels = y.detach().cpu().numpy().astype(np.float32)  # [B, C, H, W]
            
            for b in range(pred_imgs.shape[0]):
                pred_img = pred_imgs[b]  # [C, H, W]
                label = labels[b]  # [C, H, W]
                
                # PSNR计算
                scene_psnr_l = calculate_psnr(label, pred_img, data_range=1.0)
                
                # SSIM计算
                pred_img_255 = np.clip(pred_img * 255.0, 0., 255.).transpose(1, 2, 0)
                label_255 = np.clip(label * 255.0, 0., 255.).transpose(1, 2, 0)
                scene_ssim_l = calculate_ssim(pred_img_255, label_255, data_range=255.0)
                
                test_results['psnr_l'].append(scene_psnr_l)
                test_results['ssim_l'].append(scene_ssim_l)

        
        psnr_l = np.mean(test_results['psnr_l'])
        ssim_l = np.mean(test_results['ssim_l'])
        # psnr_mu = np.mean(test_results['psnr_mu'])
        # ssim_mu = np.mean(test_results['ssim_mu'])

        strt = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        log_entry = {
            "date": strt,
            "Val step": step,
            "psnr_l": psnr_l,
            "ssim_l": ssim_l
            # "psnr_mu": psnr_mu,
            # "ssim_mu": ssim_mu
        }
        
        self.save_log(self.log_file, log_entry)
        
        print(f"[{strt}] Val step: {step}, batch: {self.configs.training.batch_size}", \
            f"psnr-l: {psnr_l:.4f}", \
            f"ssim-l: {ssim_l:.4f}")
        
        self.model_sam.append((step, psnr_l, ssim_l))

        # 如果启用了 wandb，则记录日志和图像
        if self.configs.wandb.is_use_wandb:
            # 记录指标到 wandb
            wandb.log({
                "val_psnr_l": psnr_l,
                # "val_psnr_mu": psnr_mu,
                "val_ssim_l": ssim_l
                # "val_ssim_mu": ssim_mu
            })

    def find_best_model(self, date_dir):
        """
        根据 psnr_l 找到最佳模型。
        """
        if not self.model_sam:
            print("No models were evaluated during training.")
            return
        if self.args.gpu == 0:
            # 按照 psnr_l 从大到小排序
            sorted_models = sorted(self.model_sam, key=lambda x: x[1], reverse=True)

            # 找到 psnr_l 最大的模型
            best_step, best_psnr_l, best_ssim_l = sorted_models[0]
            
            log_entry = {
                "model_path": os.path.join(date_dir, f"model_{best_step}.pth"),
                "best_step": best_step,
                "best_psnr_l": best_psnr_l,
                "best_ssim_l": best_ssim_l
                # "best_psnr_mu": best_psnr_mu,
                # "best_ssim_mu": best_ssim_mu
            }
            
            self.save_log(self.log_file, log_entry)

            print(f"Best model found at step: {best_step}, at folder: {date_dir}")
            print(f"PSNR-L: {best_psnr_l:.4f}, SSIM-L: {best_ssim_l:.4f}")

    def save_log(self, log_file, log_entry, indent=None):
        if not self.configs.DEBUG:
            with open(log_file, 'a') as f:
                json.dump(log_entry, f, indent=indent)
                f.write('\n')
            
    def sync_loss(self, loss):
        """
        在分布式训练中同步张量形式的 loss 值，并返回平均值。
        如果不是分布式训练，则直接返回 loss 值。

        参数:
            loss (torch.Tensor): 当前 GPU 上的损失值（标量张量）。
            args: 包含分布式训练相关信息的对象。
            device: 当前 GPU 的设备。

        返回:
            float: 同步后的平均 loss 值。
        """
        # if dist.is_initialized() and self.args.world_size > 1:
        #     dist.barrier()
        #     # 确保 loss 是 Tensor
        #     if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
        #         raise ValueError("Loss must be a scalar tensor.")

        #     # 使用 all_reduce 汇总所有 GPU 的 loss
        #     dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        #     # 计算平均 loss
        #     loss_value = loss.item() / self.args.world_size
        # else:
        #     # 非分布式训练时直接使用 loss
        #     loss_value = loss.item()

        return loss.item()
    
if __name__ == '__main__':
    ## 测试本文件，在models文件夹所在目录下运行，“python -m models.ddm_hdr”
    ## 避免在包内运行测试代码
    
    # device = torch.device("cpu")

    # x0 = torch.randn(1, 3, 256, 256, device=device)
    # presl = torch.zeros_like(x0)

    # args = argparse.Namespace()
    # args.sampling_timesteps = 10

    # config_dict = {
    #     "data": {
    #         "conditional": True
    #     },
    #     "LP": {
    #         "num_high": 2,
    #         "in_chans": 3
    #     },
    #     "model": {
    #         "in_channels": 3,
    #         "out_ch": 3,
    #         "ch": 64,
    #         "ch_mult": [1, 2, 3, 4],
    #         "num_res_blocks": 2,
    #         "dropout": 0.0,
    #         "ema_rate": 0.999,
    #         "ema": True,
    #         "resamp_with_conv": True
    #     },
    #     "diffusion": {
    #         "beta_schedule": "linear",
    #         "beta_start": 0.0001,
    #         "beta_end": 0.02,
    #         "num_diffusion_timesteps": 200
    #     }
    # }

    # # 转换为 Namespace
    # configs = dict_to_namespace(config_dict)

    # pmodel = Net(args, configs).to(device)
    # pmodel.eval()

    # with torch.no_grad():
    #     x0, imgh, imgw = check_size(x0)
    #     presl, _, _ = check_size(presl)
        
    #     pred = pmodel(x0, presl)
    #     pred_x = pred['pred_x'][:, :, :imgh, :imgw]
        
    # print(pred['pred_x'].shape)

    lp_config = OmegaConf.load("/datadisk2/xuronghao/Projects/Basic/configs/ll.yml")
    LP = Lap_Pyramid_Conv(lp_config.VAE.LP)
    inputp = "/datadisk2/xuronghao/Datasets/GoPro/train/input/GOPR0871_11_01-000249.png"
    targetp = "/datadisk2/xuronghao/Datasets/GoPro/train/target/GOPR0871_11_01-000249.png"
    TF = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    input = TF(Image.open(inputp).convert("RGB")).unsqueeze(0)
    target = TF(Image.open(targetp).convert("RGB")).unsqueeze(0)

    
    in_lp = LP.pyramid_decom(input)
    gt_lp = LP.pyramid_decom(target)

    pred_x = LP(in_lp[:-2] + [gt_lp[-2], in_lp[-1]], input)
    print(pred_x.shape)
    