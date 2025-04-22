import argparse
import os
import yaml
import torch
import torch.backends.cudnn
import torch.utils.data
import numpy as np
import datasets
import models
import torch.distributed as dist

def parse_args_and_config_ddp():
    parser = argparse.ArgumentParser(description='Training Wavelet-Based Diffusion Model with DDP')
    # unset
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--world-size', default=4, type=int, help='number of distributed processes')
    # torchrun
    parser.add_argument('--local_rank', type=int, help='rank of distributed processes')
    #
    parser.add_argument("--config", default=r"D:\work\Diffusion-Low-Light-main\configs\SIG17_hdr.yml", type=str,
                         help="Path to the config file")
    parser.add_argument('--resume', default='', type=str,
                        help='Path for checkpoint to load and resume')
    parser.add_argument("--sampling_timesteps", type=int, default=10,
                        help="DDIM TIME INTERVAL")
    parser.add_argument("--image_folder", default='results/', type=str,
                         help="Location to save restored validation image patches")
    parser.add_argument('--seed', default=230, type=int, metavar='N',
                        help='Seed for initializing training (default: 230)')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    
    # if args.local_rank == 0 and new_config.wandb.is_use_wandb:
    #     os.environ["WANDB_API_KEY"] = new_config.wandb.APIkeys 

    # 初始化各进程环境
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return
        
    # 设置当前程序使用的GPU。根据python的机制，在单卡训练时本质上程序只使用一个CPU核心，而DataParallel
    # 不管用了几张GPU，依然只用一个CPU核心，在分布式训练中，每一张卡分别对应一个CPU核心，即几个GPU几个CPU核心
    torch.cuda.set_device(args.gpu)

    # 分布式初始化
    args.dist_url = 'env://'  # 设置url
    args.dist_backend = 'nccl'  # 通信后端，nvidia GPU推荐使用NCCL
    # print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, 
                            init_method=args.dist_url, 
                            world_size=args.world_size, 
                            rank=args.rank)
    dist.barrier()  # 等待所有进程都初始化完毕，即所有GPU都要运行到这一步以后在继续

    return args, new_config, config

def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Training Wavelet-Based Diffusion Model')
    parser.add_argument("--config", default='LOLv1.yml', type=str,
                        help="Path to the config file")
    parser.add_argument('--resume', default='', type=str,
                        help='Path for checkpoint to load and resume')
    parser.add_argument("--sampling_timesteps", type=int, default=10,
                        help="Number of implicit sampling steps for validation image patches")
    parser.add_argument("--image_folder", default='results/', type=str,
                        help="Location to save restored validation image patches")
    parser.add_argument('--seed', default=230, type=int, metavar='N',
                        help='Seed for initializing training (default: 230)')
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    # setup device to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # data loading
    print("=> using dataset '{}'".format(config.data.type))
    DATASET = datasets.__dict__[config.data.type](config)

    # create model
    print("=> creating denoising-diffusion model...")
    diffusion = models.__dict__[config.model_name](args, config)
    diffusion.train(DATASET)

def main_DDP():
    # Parse arguments
    args, configs, dict_configs = parse_args_and_config_ddp()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    
    # Data loading
    DATASET = datasets.__dict__[configs.data.type](configs)
    # 获取训练集、验证集和训练集采样器
    train_loader, val_loader, train_sampler = DATASET.get_loaders_ddp()
    
    # Model
    model = models.__dict__[configs.model_name](configs=configs, args=args, ddp=True)
    model.train(train_loader, val_loader, train_sampler, dict_configs)

if __name__ == "__main__":
    main_DDP()
