#-*- coding:utf-8 -*-  
import numpy as np
import os, glob
import cv2
import math
# import imageio
from math import log10
from PIL import Image
import random
import torch
import torch.nn as nn
import torch.nn.init as init
# from skimage.measure. import compare_psnr
# from skimage.measure import 
# from skimage.metrics import peak_signal_noise_ratio as compare_psnr
# imageio.plugins.freeimage.download()


def calculate_psnr(label, pred_img, data_range=1.0):

    """
    手动计算 PSNR。
    
    参数:
        label (numpy.ndarray): 真实图像。
        pred_img (numpy.ndarray): 预测图像。
        data_range (float): 图像的动态范围（最大值 - 最小值）。
    
    返回:
        float: PSNR 值（单位: dB)。
    """
    # 计算 MSE
    mse = np.mean((label - pred_img) ** 2)
    
    # 如果 MSE 为 0，表示两幅图像完全相同，PSNR 为无穷大
    if mse == 0:
        return float('inf')
    
    # 计算 PSNR
    psnr_value = 10 * np.log10((data_range ** 2) / mse)
    return psnr_value

def ssim(img1, img2, data_range=255):
    C1 = (0.01 * data_range)**2
    C2 = (0.03 * data_range)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2, data_range=255):
    """
    calculate SSIM

    :param img1: [0, 255]
    :param img2: [0, 255]
    :return:
    """
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2, data_range)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2, data_range))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2), data_range)
    else:
        raise ValueError('Wrong input image dimensions.')

def list_all_files_sorted(folder_name, extension=""):
    return sorted(glob.glob(os.path.join(folder_name, "*" + extension)))

def read_expo_times(file_name):
    return np.power(2, np.loadtxt(file_name))

def read_images(file_names):
    imgs = []
    for img_str in file_names:
        img = cv2.imread(img_str, -1)
        # equivalent to im2single from Matlab
        img = img / 2 ** 16
        img = np.float32(img)
        img.clip(0, 1)
        imgs.append(img)
    return np.array(imgs)

def read_label(file_path, file_name):
    label = cv2.imread(os.path.join(file_path, file_name), -1)
    # label = label[:, :, [2, 1, 0]]  ##cv2
    return label

def ldr_to_hdr(imgs, expo, gamma):
    return (imgs ** gamma) / (expo + 1e-8)

def range_compressor(x):
    return (np.log(1 + 5000 * x)) / np.log(1 + 5000)

def range_compressor_cuda(hdr_img, mu=5000):
    return (torch.log(1 + mu * hdr_img)) / math.log(1 + mu)

def range_compressor_tensor(x, device):
    a = torch.tensor(1.0, device=device, requires_grad=False)
    mu = torch.tensor(5000.0, device=device, requires_grad=False)
    return (torch.log(a + mu * x)) / torch.log(a + mu)

def psnr(x, target):
    sqrdErr = np.mean((x - target) ** 2)
    return 10 * log10(1/sqrdErr)

# def batch_psnr(img, imclean, data_range):
#     Img = img.data.cpu().numpy().astype(np.float32)
#     Iclean = imclean.data.cpu().numpy().astype(np.float32)
#     psnr = 0
#     for i in range(Img.shape[0]):
#         psnr += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
#         #peak_signal_noise_ratio(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
#     return (psnr/Img.shape[0])

# def batch_psnr_mu(img, imclean, data_range):
#     img = range_compressor_cuda(img)
#     imclean = range_compressor_cuda(imclean)
#     Img = img.data.cpu().numpy().astype(np.float32)
#     Iclean = imclean.data.cpu().numpy().astype(np.float32)
#     psnr = 0
#     for i in range(Img.shape[0]):
#         psnr += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
#         #peak_signal_noise_ratio(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
#     return (psnr/Img.shape[0])

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr * (0.5 ** (epoch // args.lr_decay_interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def init_parameters(net):
    """Init layer parameters"""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

def set_random_seed(seed):
    """Set random seed for reproduce"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# def radiance_writer(out_path, image):

#     with open(out_path, "wb") as f:
#         f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
#         f.write(b"-Y %d +X %d\n" %(image.shape[0], image.shape[1]))

#         brightest = np.maximum(np.maximum(image[...,0], image[...,1]), image[...,2])
#         mantissa = np.zeros_like(brightest)
#         exponent = np.zeros_like(brightest)
#         np.frexp(brightest, mantissa, exponent)
#         scaled_mantissa = mantissa * 255.0 / brightest
#         rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
#         rgbe[...,0:3] = np.around(image[...,0:3] * scaled_mantissa[...,None])
#         rgbe[...,3] = np.around(exponent + 128)

#         rgbe.flatten().tofile(f)

def radiance_writer(out_path, image):
    """
    将 HDR 图像保存为 Radiance RGBE 格式。
    
    参数:
        out_path (str): 输出文件路径。
        image (numpy.ndarray): 输入 HDR 图像，形状为 (H, W, 3)，数据范围通常为 [0, ∞)。
    """
    # 确保输入图像是浮点类型
    image = image.astype(np.float32)
    
    # 打开文件并写入头部信息
    with open(out_path, "wb") as f:
        f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" % (image.shape[0], image.shape[1]))
        
        # 计算每个像素的最大亮度值 (brightest)
        brightest = np.maximum(np.maximum(image[..., 0], image[..., 1]), image[..., 2])
        
        # 防止 brightest 中出现零值（避免除以零）
        brightest = np.maximum(brightest, 1e-6)
        
        # 使用 np.frexp 分解 brightest 为尾数 (mantissa) 和指数 (exponent)
        mantissa, exponent = np.frexp(brightest)
        
        # 计算 scaled_mantissa
        scaled_mantissa = mantissa * 255.0 / brightest
        
        # 检查并替换 NaN 和 Inf
        scaled_mantissa = np.nan_to_num(scaled_mantissa, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 初始化 rgbe 数组
        rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        
        # 计算 RGBE 的 RGB 部分
        rgbe[..., 0:3] = np.around(image[..., 0:3] * scaled_mantissa[..., None]).astype(np.uint8)
        
        # 计算 RGBE 的 E 部分
        rgbe[..., 3] = np.around(exponent + 128).astype(np.uint8)
        
        # 将 rgbe 数据写入文件
        rgbe.flatten().tofile(f)

def save_hdr(path, image):
    return radiance_writer(path, image)

def range_compressor(x):
    return (np.log(1 + 5000 * x)) / np.log(1 + 5000)





