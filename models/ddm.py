import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import utils
# from caffe2.contrib.tensorboard.tensorboard import Config
from models.unet import DiffusionUNet
from models.wavelet import DWT, IWT
from pytorch_msssim import ssim
from models.mods import HFRM
from .hdr_transformer import HDRTransformer
from hdrutils.utils import radiance_writer, range_compressor
import cv2

def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


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
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
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
        self.device = config.device

        num_in_ch = config.hdrvit.in_chans
        num_out_ch = config.hdrvit.out_chans
        ################################### 1. Feature Extraction Network ###################################
        # coarse feature
        self.conv_f1 = nn.Conv2d(num_in_ch, config.hdrvit.embed_dim, 3, 1, 1)
        self.conv_f2 = nn.Conv2d(num_in_ch, config.hdrvit.embed_dim, 3, 1, 1)
        self.conv_f3 = nn.Conv2d(num_in_ch, config.hdrvit.embed_dim, 3, 1, 1)
        # spatial attention module
        self.att_module_l = SpatialAttentionModule(config.hdrvit.embed_dim)
        self.att_module_h = SpatialAttentionModule(config.hdrvit.embed_dim)
        self.conv_first = nn.Conv2d(config.hdrvit.embed_dim * 3, num_out_ch, 3, 1, 1)
        ###################################
        self.high_enhance0 = HFRM(in_channels=3, out_channels=64)
        self.high_enhance1 = HFRM(in_channels=3, out_channels=64)
        self.Unet = DiffusionUNet(config)

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = self.betas.shape[0]

    @staticmethod
    def compute_alpha(beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def sample_training(self, x_cond, b, eta=0.):
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        n, c, h, w = x_cond.shape
        seq_next = [-1] + list(seq[:-1])
        x = torch.randn(n, c, h, w, device=self.device)
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

    def forward(self, x1, x2, x3, label):
        ########################
        f1 = self.conv_f1(x1)
        f2 = self.conv_f2(x2)
        f3 = self.conv_f3(x3)

        # spatial feature attention 
        f1_att_m = self.att_module_h(f1, f2)
        f1_att = f1 * f1_att_m
        f3_att_m = self.att_module_l(f3, f2)
        f3_att = f3 * f3_att_m
        x_in = self.conv_first(torch.cat((f1_att, f2, f3_att), dim=1))
     
        data_dict["x"] = torch.cat([x_in, label], dim=1)
        x = torch.cat([x_in, label], dim=1)

        ########################
        data_dict = {}
        dwt, idwt = DWT(), IWT()
        #
        input_img = x[:, :3, :, :]
        n, c, h, w = input_img.shape
        input_img_norm = data_transform(input_img)
        input_dwt = dwt(input_img_norm)

        input_LL, input_high0 = input_dwt[:n, ...], input_dwt[n:, ...]

        input_high0 = self.high_enhance0(input_high0)

        input_LL_dwt = dwt(input_LL)
        input_LL_LL, input_high1 = input_LL_dwt[:n, ...], input_LL_dwt[n:, ...]
        input_high1 = self.high_enhance1(input_high1)

        b = self.betas.to(input_img.device)

        t = torch.randint(low=0, high=self.num_timesteps, size=(input_LL_LL.shape[0] // 2 + 1,)).to(self.device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:input_LL_LL.shape[0]].to(x.device)
        a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)

        e = torch.randn_like(input_LL_LL)

        if self.training:
            gt_img_norm = data_transform(x[:, 3:, :, :])
            gt_dwt = dwt(gt_img_norm)
            gt_LL, gt_high0 = gt_dwt[:n, ...], gt_dwt[n:, ...]

            gt_LL_dwt = dwt(gt_LL)
            gt_LL_LL, gt_high1 = gt_LL_dwt[:n, ...], gt_LL_dwt[n:, ...]

            x = gt_LL_LL * a.sqrt() + e * (1.0 - a).sqrt()
            noise_output = self.Unet(torch.cat([input_LL_LL, x], dim=1), t.float())
            denoise_LL_LL = self.sample_training(input_LL_LL, b)

            pred_LL = idwt(torch.cat((denoise_LL_LL, input_high1), dim=0))

            pred_x = idwt(torch.cat((pred_LL, input_high0), dim=0))
            pred_x = inverse_data_transform(pred_x)

            data_dict["input_high0"] = input_high0
            data_dict["input_high1"] = input_high1
            data_dict["gt_high0"] = gt_high0
            data_dict["gt_high1"] = gt_high1
            data_dict["pred_LL"] = pred_LL
            data_dict["gt_LL"] = gt_LL
            data_dict["noise_output"] = noise_output
            data_dict["pred_x"] = pred_x
            data_dict["e"] = e

        else:
            denoise_LL_LL = self.sample_training(input_LL_LL, b)
            pred_LL = idwt(torch.cat((denoise_LL_LL, input_high1), dim=0))
            pred_x = idwt(torch.cat((pred_LL, input_high0), dim=0))
            pred_x = inverse_data_transform(pred_x)

            data_dict["pred_x"] = pred_x

        return data_dict


class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model = Net(args, config)
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)

        # input: x0 x1 x2 (6) -> x(3)
        # self.hdrvit_model = HDRTransformer(in_chans=config.hdrvit.in_chans,
        #                             out_chans=config.hdrvit.out_chans,
        #                             embed_dim=config.hdrvit.embed_dim)#.to(self.device)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.TV_loss = TVLoss()

        self.optimizer, self.scheduler = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint {} step {}".format(load_path, self.step))

    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()

        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch)
            data_start = time.time()
            data_time = 0
            # for i, (x, y) in enumerate(train_loader):
            for i, data_dict in enumerate(train_loader):
                # x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                input0 = data_dict["input0"].to(self.device)
                input1 = data_dict["input1"].to(self.device)
                input2 = data_dict["input2"].to(self.device)
                label = data_dict["label"].to(self.device)
                data_time += time.time() - data_start

                # self.hdrvit_model.train()
                self.model.train()
                self.step += 1

                #x = x.to(self.device)
                # x_in = self.hdrvit_model(input0, input1, input2)
                # x_in.to(self.device)
                # x = torch.cat([x_in, label], dim=1)

                output = self.model(input0, input1, input2, label)

                noise_loss, photo_loss, frequency_loss = self.estimation_loss(output['x'], output)
                loss = noise_loss + photo_loss + frequency_loss

                if self.step % 10 == 0:
                    print("step:{}, lr:{:.6f}, noise_loss:{:.4f}, photo_loss:{:.4f}, "
                          "frequency_loss:{:.4f}".format(self.step, self.scheduler.get_last_lr()[0],
                                                         noise_loss.item(), photo_loss.item(),
                                                         frequency_loss.item()))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()

                if self.step % self.config.training.validation_freq == 0 and self.step != 0:
                    self.model.eval()
                    self.sample_validation_patches(val_loader, self.step)

                    utils.logging.save_checkpoint({'step': self.step, 'epoch': epoch + 1,
                                                   'state_dict': self.model.state_dict(),
                                                   'optimizer': self.optimizer.state_dict(),
                                                   'scheduler': self.scheduler.state_dict(),
                                                   'ema_helper': self.ema_helper.state_dict(),
                                                   'params': self.args,
                                                   'config': self.config},
                                                    filename=os.path.join(self.config.data.ckpt_dir, 'model_latest'))
                        
            self.scheduler.step()

    def estimation_loss(self, x, output):

        input_high0, input_high1, gt_high0, gt_high1 = output["input_high0"], output["input_high1"],\
                                                       output["gt_high0"], output["gt_high1"]

        pred_LL, gt_LL, pred_x, noise_output, e = output["pred_LL"], output["gt_LL"], output["pred_x"],\
                                                  output["noise_output"], output["e"]

        gt_img = x[:, 3:, :, :].to(self.device)
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
        ssim_loss = 1 - ssim(pred_x, gt_img, data_range=1.0).to(self.device)

        photo_loss = content_loss + ssim_loss

        return noise_loss, photo_loss, frequency_loss

    def check_size(self, x):
        _, _, img_h, img_w = x.shape
        img_h_32 = int(32 * np.ceil(img_h / 32.0))
        img_w_32 = int(32 * np.ceil(img_w / 32.0))
        x = F.pad(x, (0, img_w_32 - img_w, 0, img_h_32 - img_h), 'reflect')
        return x, img_h, img_w
    
    def sample_validation_patches(self, val_loader, step):
        image_folder = os.path.join(self.args.image_folder, self.config.data.type + str(self.config.data.val.patch_size))
        self.model.eval()
        with torch.no_grad():
            print(f"Processing a single batch of validation images at step: {step}")
            # for i, (x, y) in enumerate(val_loader):
            for i, data_dict in enumerate(val_loader):

                input0, img_h, img_w = self.check_size(data_dict["input0"])
                input1, _, _ = self.check_size(data_dict["input1"])
                input2, _, _ = self.check_size(data_dict["input2"])
                label, _, _ = self.check_size(data_dict["label"])

                # x_in = self.hdrvit_model(input0, input1, input2)
                # x = torch.cat([x_in, label], dim=1)

                out = self.model(input0.to(self.device), input1.to(self.device), input2.to(self.device), label.to(self.device))

                pred_x = out["pred_x"]
                pred_x = pred_x[:, :, :img_h, :img_w]

                pred_x = torch.squeeze(pred_x.detach().cpu()).numpy().astype(np.float32)
                save_pred_img = (range_compressor(pred_x) * 255).astype(np.uint8)
                save_pred_img = cv2.cvtColor(save_pred_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(image_folder, str(step), f"{i}.png"), save_pred_img)

                save_pred_hdr = pred_x.copy().transpose(1, 2, 0)[..., ::-1]
                radiance_writer(os.path.join(image_folder, str(step), f"{i}.hdr"), save_pred_hdr)


