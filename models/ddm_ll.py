import argparse
import datetime
import json
import re
import time, os
import numpy as np
import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F
from models.unet import DiffusionUNet
from models.wavelet import DWT, IWT
from pytorch_msssim import ssim
from models.mods import HFRM
from hdrutils.utils import radiance_writer, range_compressor, calculate_psnr, calculate_ssim
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.optimize import get_optimizer
import wandb
from tqdm import tqdm

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
    # shadow_param = (1 − μ)⋅current_param + μ⋅shadow_param
    # 帮助生成更稳定的模型权重，提升模型的泛化能力和鲁棒性。
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel) or isinstance(module, nn.parallel.DistributedDataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel) or isinstance(module, nn.parallel.DistributedDataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel) or isinstance(module, nn.parallel.DistributedDataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
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
    def __init__(self, num_high=3, in_chans=24):
        super(Lap_Pyramid_Conv, self).__init__()

        self.num_high = num_high
        self.channels = in_chans
        self.kernel = nn.Parameter(self.gauss_kernel(channels=self.channels), requires_grad=False)
        
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
        current = img
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
            pyr.append((diff, current))
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1] # pyr=[h_0^hat, h_1^hat, h_2^hat, I_3^hat]
        # print("***********************")
        # print(image.size())
        # print(len(pyr[:-1]))
        for level in reversed(pyr[:-1]):
            # https://www.jianshu.com/p/e7de4cd92f68
            up = self.upsample(image)
            # if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
            if up.shape != level.shape:
                # up = nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
                up = F.interpolate(up, size=(level.shape[2], level.shape[3]))
            image = up + level
        return image


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

        self.lp = Lap_Pyramid_Conv(num_high=config.LP.num_high, in_chans=config.LP.in_chans)
        # self.conv_LL_LL = nn.Sequential(
        #     nn.Conv2d(6, 6, 3, 1, 1),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(6, 6, 3, 1, 1),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(6, 3, 3, 1, 1)
        # )
        # self.conv_LL = nn.Sequential(
        #     nn.Conv2d(6, 6, 3, 1, 1),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(6, 6, 3, 1, 1),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(6, 3, 3, 1, 1)
        # )
        # self.final_conv = nn.Conv2d(3, 3, 3, 1, 1)

        # num_in_ch = config.hdrvit.in_chans
        # num_out_ch = config.hdrvit.out_chans
        # ################################### 1. Feature Extraction Network ###################################
        # # coarse feature
        # self.conv_f1 = nn.Conv2d(num_in_ch, config.hdrvit.embed_dim, 3, 1, 1)
        # self.conv_f2 = nn.Conv2d(num_in_ch, config.hdrvit.embed_dim, 3, 1, 1)
        # self.conv_f3 = nn.Conv2d(num_in_ch, config.hdrvit.embed_dim, 3, 1, 1)
        # # spatial attention module
        # self.att_module_l = SpatialAttentionModule(config.hdrvit.embed_dim)
        # self.att_module_h = SpatialAttentionModule(config.hdrvit.embed_dim)
        # self.conv_first = nn.Conv2d(config.hdrvit.embed_dim * 3, num_out_ch, 3, 1, 1)
        ####################################################################################################
        self.high_enhance0 = HFRM(in_channels=3, out_channels=64)
        self.high_enhance1 = HFRM(in_channels=3, out_channels=64)
        self.Unet = DiffusionUNet(config)

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
    
    def recon(self, input_lp, denoise_high, x):
        assert (input_lp[1][0].shape == denoise_high.shape), print("input_lp[1][0].shape != denoise_high.shape")
        
        # cat(input_LL_LL, denoise_high1)
        # up -> LL
        # cat(input_high0, LL)
        # conv3
        
        input_high0, input_LL = input_lp[0]
        input_high1, input_LL_LL = input_lp[1]
        input_down =  input_lp[-1]
        # input_denoise_LL_LL = torch.cat([input_LL_LL, denoise_high], dim=1)
        # pred_LL = input_LL_LL + denoise_high
        # pred_LL += input_LL_LL

        # pred_LL = F.interpolate(pred_LL, scale_factor=2, mode='bicubic', align_corners=False)
        
        # pred_x = torch.cat([input_high0, pred_LL], dim=1)

        pred_x = self.lp.pyramid_recons([input_high0, denoise_high, input_down])

        # pred_x = self.conv_LL(pred_x)
        # pred_x += self.final_conv(x)

        return pred_x, None

    def forward(self, x, label):
        data_dict = {}
        ########################
        input_lp = self.lp.pyramid_decom(x)
        input_high0, input_LL = input_lp[0]
        input_high1, input_LL_LL = input_lp[1] 
        ########################
        b = self.betas.to(x.device)

        # 对称时间步序列：结合了随机采样和对称补集的优点，能够覆盖正向和反向扩散的不同阶段
        t = torch.randint(low=0, high=self.num_timesteps, size=(input_high1.shape[0] // 2 + 1,)).to(x.device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:input_high1.shape[0]].to(x.device)
        a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)

        e = torch.randn_like(input_high1)

        if self.training:
            gt_lp= self.lp.pyramid_decom(label)
            gt_high0, gt_LL = gt_lp[0]
            gt_high1, gt_LL_LL = gt_lp[1]  

            xet = gt_high1 * a.sqrt() + e * (1.0 - a).sqrt()
            noise_output = self.Unet(torch.cat([input_high1, xet], dim=1), t.float())
            denoise_high1 = self.sample_training(input_high1, b)

            # cat(input_LL_LL, denoise_high1)
            # up -> LL
            # cat(input_high0, LL)
            # conv3
            #
            # denoise_high1 = inverse_data_transform(denoise_high1)
            pred_x, pred_LL = self.recon(input_lp, denoise_high1, x)
            # cl_pred_x = torch.clamp(pred_x, min=0, max=1)

            data_dict["input_lp"] = input_lp               
            data_dict["gt_lp"] = gt_lp
            data_dict["pred_high1"] = denoise_high1

            data_dict["noise_output"] = noise_output
            data_dict["pred_x"] = pred_x
            data_dict["pred_LL"] = pred_LL
            data_dict["gt"] = label 
            data_dict["e"] = e

        else:
            # label 随意填充为 zeros_like(input_img) 
            denoise_high1 = self.sample_training(input_high1, b)
            pred_x, _ = self.recon(input_lp, denoise_high1, x)
            # cl_pred_x = torch.clamp(pred_x, min=0, max=1)
            data_dict["pred_x"] = pred_x

        return data_dict
    
class LL_DDM(nn.Module):
    def __init__(self, configs, args, ddp=True):
        super().__init__()
        self.configs = configs
        self.args = args
        self.model = Net(config=configs, args=args)
        if ddp:
            self.model = DDP(self.model.to(self.args.gpu), 
                            device_ids=[self.args.gpu],
                            find_unused_parameters=True)
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
                                'state_dict': self.model.module.state_dict(),  # 保存 .module 的状态字典
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
                save_pred_img = cv2.cvtColor(save_pred_img.transpose(1, 2, 0), cv2.COLOR_BGR2RGB)

                # 保存/记录图像（每20个batch保存一次）
                # if not self.configs.DEBUG and i % 20 == 0:
                #     cv2.imwrite(os.path.join(image_folder, f"{step}_{i}_{b}.png"), save_pred_img)
                
                if self.configs.wandb.is_use_wandb and i % 20 == 0:
                    wandb.log({
                        f"HDR_MU_IMG_{i}_{b}": wandb.Image(save_pred_img, caption=f"val_{step}_{i}_{b}")
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

            # pred_x = torch.squeeze(pred_x_tensor.detach().cpu()).numpy().astype(np.float32)
            # save_pred_img = (pred_x * 255).astype(np.uint8)
            # save_pred_img = cv2.cvtColor(save_pred_img.transpose(1, 2, 0), cv2.COLOR_BGR2RGB)

            # if not self.configs.DEBUG and i % 20 == 0:
            #     cv2.imwrite(os.path.join(image_folder, f"{step}_{i}.png"), save_pred_img)
            # # 上传预测图像到 wandb
            # if self.configs.wandb.is_use_wandb and i % 20 == 0:
            #     wandb.log({
            #         f"HDR_MU_IMG_{i}": wandb.Image(save_pred_img, caption=f"val_{step}")
            #     })
                
            # # save_hdr = pred_x.copy().transpose(1, 2, 0)[..., ::-1]
            # # if not self.configs.DEBUG:
            # #     radiance_writer(os.path.join(image_folder, f"{step}_{i}.hdr"), save_hdr)
            
            # ## psnr
            # pred_img = torch.squeeze(out["pred_x"].detach().cpu()).numpy().astype(np.float32)

            # label = torch.squeeze(y.detach().cpu()).numpy().astype(np.float32)
            # scene_psnr_l = calculate_psnr(label, pred_img, data_range=1.0)
            # label_mu = range_compressor(label)
            # # pred_img_mu = range_compressor(pred_img) # the estimated tonemapped HDR image
            # # scene_psnr_mu = calculate_psnr(label_mu, pred_img_mu, data_range=1.0)
            
            # # ssim-l
            # pred_img = np.clip(pred_img * 255.0, 0., 255.).transpose(1, 2, 0)
            # label = np.clip(label * 255.0, 0., 255.).transpose(1, 2, 0)
            # scene_ssim_l = calculate_ssim(pred_img, label, data_range=255.0)
            
            # # ssim-\mu
            # # pred_img_mu = np.clip(pred_img_mu * 255.0, 0., 255.).transpose(1, 2, 0)
            # # label_mu = np.clip(label_mu * 255.0, 0., 255.).transpose(1, 2, 0)
            # # scene_ssim_mu = calculate_ssim(pred_img_mu, label_mu, data_range=255.0)
        
            # test_results['psnr_l'].append(scene_psnr_l)
            # test_results['ssim_l'].append(scene_ssim_l)
            # # test_results['psnr_mu'].append(scene_psnr_mu)
            # # test_results['ssim_mu'].append(scene_ssim_mu)
        
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
    
    device = torch.device("cpu")

    x0 = torch.randn(1, 3, 256, 256, device=device)
    presl = torch.zeros_like(x0)

    args = argparse.Namespace()
    args.sampling_timesteps = 10

    config_dict = {
        "data": {
            "conditional": True
        },
        "LP": {
            "num_high": 2,
            "in_chans": 3
        },
        "model": {
            "in_channels": 3,
            "out_ch": 3,
            "ch": 64,
            "ch_mult": [1, 2, 3, 4],
            "num_res_blocks": 2,
            "dropout": 0.0,
            "ema_rate": 0.999,
            "ema": True,
            "resamp_with_conv": True
        },
        "diffusion": {
            "beta_schedule": "linear",
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "num_diffusion_timesteps": 200
        }
    }

    # 转换为 Namespace
    configs = dict_to_namespace(config_dict)

    pmodel = Net(args, configs).to(device)
    pmodel.eval()

    with torch.no_grad():
        x0, imgh, imgw = check_size(x0)
        presl, _, _ = check_size(presl)
        
        pred = pmodel(x0, presl)
        pred_x = pred['pred_x'][:, :, :imgh, :imgw]
        
    print(pred['pred_x'].shape)
    
    
    
    