import os
import torch
import torch.utils.data
import PIL
from PIL import Image
import re
from datasets.data_augment import PairCompose, PairRandomCrop, PairToTensor
from torch.utils.data import Dataset, DataLoader

class LLdataset:
    def __init__(self, config):
        self.config = config

    def get_loaders(self):

        # train_dataset = AllWeatherDataset(os.path.join(self.config.data.data_dir, self.config.data.train_dataset, 'train'),
        #                                   patch_size=self.config.data.patch_size,
        #                                   filelist='{}_train.txt'.format(self.config.data.train_dataset))
        # val_dataset = AllWeatherDataset(os.path.join(self.config.data.data_dir, self.config.data.val_dataset, 'val'),
        #                                 patch_size=self.config.data.patch_size,
        #                                 filelist='{}_val.txt'.format(self.config.data.val_dataset), train=False)

        train_dataset = AllWeatherDataset(self.config.data.data_dir,
                                        patch_size=self.config.data.patch_size,
                                        filelist='{}_train.txt'.format(self.config.data.train_dataset))

        val_dataset = AllWeatherDataset(self.config.data.data_dir,
                                        patch_size=self.config.data.patch_size,
                                        filelist='{}_val.txt'.format(self.config.data.val_dataset), train=False)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                                 num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader

    def get_loaders_ddp(self):
        # Datasets → DistributedSampler → BatchSampler → DataLoader
        train_dataset = AllWeatherDataset(
            self.config.data.data_dir,
            patch_size=self.config.data.patch_size,
            filelist='{}_train.txt'.format(self.config.data.train_dataset))
        val_dataset = AllWeatherDataset(self.config.data.data_dir,
                                        patch_size=self.config.data.patch_size,
                                        filelist='{}_val.txt'.format(self.config.data.val_dataset), train=False)
        # 2. DistributedSampler
        # 给每个rank对应的进程分配训练的样本索引，比如一共800样本8张卡，那么每张卡对应分配100个样本
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

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
                                  pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=1,
                                  #sampler=val_sampler,
                                  num_workers=self.config.data.num_workers,
                                  pin_memory=True)

        return train_loader, val_loader, train_sampler


class AllWeatherDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, filelist=None, train=True):
        super().__init__()

        self.dir = dir
        self.train = train
        self.file_list = filelist
        self.train_list = os.path.join(dir, self.file_list)

        files = os.listdir(os.path.join(self.dir,"high"))
        self.input_names = [os.path.join( "low", file )for file in files]
        self.gt_names = [os.path.join("high", file )for file in files]

        # with open(self.train_list) as f:
        #     contents = f.readlines()
        #     input_names = [i.strip() for i in contents]
        #     gt_names = [i.strip().replace('low', 'high') for i in input_names]

        # self.input_names = input_names
        # self.gt_names = gt_names
        self.patch_size = patch_size
        if self.train:
            self.transforms = PairCompose([
                PairRandomCrop(self.patch_size),
                PairToTensor()
            ])
        else:
            self.transforms = PairCompose([
                PairToTensor()
            ])

    def get_images(self, index):
        input_name = self.input_names[index].replace('\n', '')
        # print(input_name)
        gt_name = self.gt_names[index].replace('\n', '')
        img_id = re.split('/', input_name)[-1][:-4]
        input_img = Image.open(os.path.join(self.dir, input_name)) if self.dir else PIL.Image.open(input_name)
        gt_img = Image.open(os.path.join(self.dir, gt_name)) if self.dir else PIL.Image.open(gt_name)

        input_img, gt_img = self.transforms(input_img, gt_img)

        return torch.cat([input_img, gt_img], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
