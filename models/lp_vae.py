import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
    # pyramid_recons
    def pyramid_recon(self, pyr):
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
            image = up + level
        return image


class AutoencoderKL(nn.Module):
    def __init__(self, args, configs):
        super(AutoencoderKL, self).__init__()
        self.args = args
        self.configs = configs

        if self.configs.VAE.type == "LP":
            self.vae = Lap_Pyramid_Conv(num_high=self.configs.VAE.num_high, in_chans=self.configs.VAE.in_chans)
        
