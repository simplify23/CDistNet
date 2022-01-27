import os
import time
import codecs
import pickle
import h5py
import numpy as np
from PIL import Image
from PIL import ImageFile
import argparse
from tqdm import tqdm
from mmcv import Config
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class NRTRDataset_hdf5(Dataset):
    def __init__(self, hdf5_file, transform=None):
        self.data = dict()
        self._transform = transform
        self.hdf5_file = hdf5_file

    def __len__(self):
        with h5py.File(self.hdf5_file, 'r') as data:
            lens = len(data['label'])
        return lens

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_file, 'r') as data:
            image = data['image'][idx]
            image = torch.from_numpy(image)
            image = image.to(torch.float32)
            target = data['label'][idx]
            target = torch.from_numpy(target)
            target = target.to(torch.int64)
        return image, target


def make_data_loader(cfg, is_train=True):
    dataset = NRTRDataset_hdf5(
        hdf5_file=cfg.train.hdf5 if is_train else cfg.val.hdf5,
    )
    dataloader = DataLoaderX(
        dataset=dataset,
        batch_size=cfg.train.batch_size if is_train else cfg.val.batch_size,
        shuffle=True if is_train else False,
        num_workers=cfg.train.num_worker if is_train else cfg.val.num_worker,
        pin_memory=False,
    )
    return dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train NRTR')
    parser.add_argument('--config', type=str, help='train config file path')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    data_loader = make_data_loader(cfg)
    for _ in range(1):
        for idx, batch in enumerate(tqdm(data_loader)):
            print(batch[0].shape)
            # print(batch[1].shape)
            # if idx == 10:
            #     break
            # print(image.shape)
            # print(batch)

