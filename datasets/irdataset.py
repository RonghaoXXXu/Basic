import os
import cv2
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class IRdataset:
    def __init__(self, config):
        self.config = config

    def get_loaders(self):

        # train_dataset = AllWeatherDataset(os.path.join(self.config.data.data_dir, self.config.data.train_dataset, 'train'),
        #                                   patch_size=self.config.data.patch_size,
        #                                   filelist='{}_train.txt'.format(self.config.data.train_dataset))
        # val_dataset = AllWeatherDataset(os.path.join(self.config.data.data_dir, self.config.data.val_dataset, 'val'),
        #                                 patch_size=self.config.data.patch_size,
        #                                 filelist='{}_val.txt'.format(self.config.data.val_dataset), train=False)

        train_dataset = AllWeatherDataset(self.config.data.train, Training=True)
        val_dataset = AllWeatherDataset(self.config.data.val, Training=False)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                                 num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader

    def get_loaders_ddp(self):
        # Datasets → DistributedSampler → BatchSampler → DataLoader
        train_dataset = AllWeatherDataset(self.config.data.train, Training=True)
        val_dataset = AllWeatherDataset(self.config.data.val, Training=False)
        # 2. DistributedSampler
        # 给每个rank对应的进程分配训练的样本索引，比如一共800样本8张卡，那么每张卡对应分配100个样本
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)

        # 3. BatchSampler
        # 刚才每张卡分了100个样本，假设BatchSize=16，那么能分成100/16=6...4，即多出4个样本
        # 下面的drop_last=True表示舍弃这四个样本，False将剩余4个样本为一组（注意最后就不是6个一组了）
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, 
                                                            self.config.training.batch_size, 
                                                            drop_last=True)
        # 4. DataLoader
        # 不要使用shuffle参数，因为sampler已经打乱了数据顺序
        # 验证集没有采用batchsampler,因此在dataloader中使用batch_size参数即可
        train_loader = DataLoader(train_dataset, 
                                  batch_sampler=train_batch_sampler,
                                  num_workers=self.config.data.num_workers,
                                  pin_memory=True,
                                  shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=self.config.training.val_batch_size,
                                  sampler=val_sampler,
                                  num_workers=self.config.data.num_workers,
                                  pin_memory=True,
                                  shuffle=False)
        
        # assert False, print(f"{val_dataset.dir_gt}")

        return train_loader, val_loader, train_sampler


class AllWeatherDataset(Dataset):
    def __init__(self, dir, Training=True):
        super().__init__()

        self.dir = dir
        self.dir_gt = dir.dataroot_gt
        self.dir_lq = dir.dataroot_lq

        self.input_names = [os.path.join(self.dir_lq, file)for file in os.listdir(os.path.join(self.dir_lq))]
        self.gt_names = [os.path.join(self.dir_gt, file)for file in os.listdir(os.path.join(self.dir_gt))]

        # if not Training:
        #     with open("/datadisk2/xuronghao/val_input.txt", "w") as f:
        #         f.write("\n".join(self.input_names))
        #     print(f"Val Input names saved")

    def to_bgr_tensor(self, img):
        img = (img.astype(np.float32) / 255.0)
        # BGR->RGB
        img = img[:, :, [2, 1, 0]]
        # 
        return torch.from_numpy(img.transpose(2, 0, 1))
    
    def get_images(self, index):
        input_name = self.input_names[index]
        # print(input_name)
        gt_name = self.gt_names[index]

        img_id = input_name

        # img BGR
        input_img = cv2.imread(input_name)
        gt_img = cv2.imread(gt_name)

        # Normlaize
        input_t = self.to_bgr_tensor(input_img)
        gt_t = self.to_bgr_tensor(gt_img)

        return torch.cat([input_t, gt_t], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
