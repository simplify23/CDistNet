import os
import re
import time
import copy
import codecs
import pickle
import numpy as np
import six
from PIL import Image
from PIL import ImageFile
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.transforms import functional as F
from torchvision import transforms
import argparse
from mmcv import Config
from tqdm import tqdm
import lmdb

from cdistnet.data.transform import CVGeometry, CVColorJitter, CVDeterioration

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_vocab(vocab=None, vocab_size=None):
    """
    Load vocab from disk. The fisrt four items in the vocab should be <PAD>, <UNK>, <S>, </S>
    """
    # print('Load set vocabularies as %s.' % vocab)
    vocab = [' ' if len(line.split()) == 0 else line.split()[0] for line in codecs.open(vocab, 'r', 'utf-8')]
    vocab = vocab[:vocab_size]
    assert len(vocab) == vocab_size
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.resize(image, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class Compose(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, target):
        for t in self.transform:
            image, target = t(image, target)
        return image, target


class TXTDataset(Dataset):

    def __init__(
            self,
            image_dir,
            gt_file,
            word2idx,
            idx2word,
            size=(100, 32),
            max_width=256,
            rgb2gray=True,
            keep_aspect_ratio=False,
            is_test=False,
            is_lower=False,
            data_aug=True,
    ):
        self.image_dir = image_dir
        self.gt = dict()
        self.idx2word = idx2word
        self.word2idx = word2idx
        self.is_test = is_test
        self.rgb2gray = rgb2gray
        self.width = size[0]
        self.height = size[1]
        self.keep_aspect_ratio = keep_aspect_ratio
        self.max_width = max_width
        self.is_lower = is_lower
        self.data_aug = data_aug
        print("preparing data ...")
        print("path:{}".format(gt_file))
        with open(gt_file, 'r', encoding='UTF-8-sig') as f:
            all = f.readlines()
            for each in tqdm(all):
                each = each.strip().split(' ')
                image_name, text = each[0], ' '.join(each[1:])
                self.gt[image_name] = text
        self.data = list(self.gt.items())
        if self.is_test==False and self.data_aug:
            self.augment_tfs = transforms.Compose([
                CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15), distortion=0.5, p=0.5),
                CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
                CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25)
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # self.data=[(img_path,text),...]
        image_name = self.data[idx][0]
        image_path = os.path.join(self.image_dir, image_name)
        # print(image_path)
        if self.rgb2gray:
            image = Image.open(image_path).convert('L')
        else:
            image = Image.open(image_path).convert('RGB')
        h, w = image.height, image.width
        if self.is_test==False and self.data_aug:
            image = self.augment_tfs(image)
        if self.is_test:
            if h / w > 2:
                image1 = image.rotate(90, expand=True)
                image2 = image.rotate(-90, expand=True)
            else:
                image1 = copy.deepcopy(image)
                image2 = copy.deepcopy(image)
        if self.keep_aspect_ratio:
            h, w = image.height, image.width
            ratio = w / h
            image = image.resize(
                (min(max(int(self.height * ratio), self.height), self.max_width), self.height),
                Image.ANTIALIAS
            )
            if self.is_test:
                h, w = image1.height, image1.width
                ratio = w / h
                image1 = image1.resize(
                    (min(max(int(self.height * ratio), self.height), self.max_width), self.height),
                    Image.ANTIALIAS
                )
                image2 = image2.resize(
                    (min(max(int(self.height * ratio), self.height), self.max_width), self.height),
                    Image.ANTIALIAS
                )
        else:
            image = image.resize((self.width, self.height), Image.ANTIALIAS)
            if self.is_test:
                image1 = image1.resize((self.width, self.height), Image.ANTIALIAS)
                image2 = image2.resize((self.width, self.height), Image.ANTIALIAS)
        image = np.array(image)
        if self.is_test:
            image1 = np.array(image1)
            image2 = np.array(image2)
        if self.rgb2gray:
            image = np.expand_dims(image, -1)
            if self.is_test:
                image1 = np.expand_dims(image1, -1)
                image2 = np.expand_dims(image2, -1)
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32) / 128. - 1.
        if self.is_test:
            image1 = image1.transpose((2, 0, 1))
            image2 = image2.transpose((2, 0, 1))
            image1 = image1.astype(np.float32) / 128. - 1.
            image2 = image2.astype(np.float32) / 128. - 1.
            image_final = np.concatenate([image, image1, image2], axis=0)

        text = self.data[idx][1]
        if self.is_lower:
            text = [self.word2idx.get(ch.lower(), 1) for ch in text]
        else:
            text = [self.word2idx.get(ch, 1) for ch in text]
        text.insert(0, 2)
        text.append(3)
        target = np.array(text)
        if self.is_test:
            return image_final, self.gt[image_name], image_name
        return image, target

class LMDBDataset(Dataset):

    def __init__(
            self,
            image_dir,
            gt_file,
            word2idx,
            idx2word,
            size=(100, 32),
            max_width=256,
            rgb2gray=True,
            keep_aspect_ratio=False,
            is_test=False,
            is_lower=False,
            data_aug=True,
    ):
        self.image_dir = image_dir
        self.gt = dict()
        self.idx2word = idx2word
        self.word2idx = word2idx
        self.is_test = is_test
        self.rgb2gray = rgb2gray
        self.width = size[0]
        self.height = size[1]
        self.keep_aspect_ratio = keep_aspect_ratio
        self.max_width = max_width
        self.is_lower = is_lower
        self.data_aug = data_aug

        print("preparing data ...")
        print("path:{}".format(gt_file))
        self.env = lmdb.open(str(gt_file), readonly=True, lock=False, readahead=False, meminit=False)
        assert self.env, f'Cannot open LMDB dataset from {gt_file}.'
        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('num-samples'.encode()))
        print("samples = {}".format(self.length))

        if self.is_test==False and self.data_aug:
            self.augment_tfs = transforms.Compose([
                CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15), distortion=0.5, p=0.5),
                CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
                CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25)
            ])

    def __len__(self):
        return self.length

    def get(self,idx):
        with self.env.begin(write=False) as txn:
            image_key, label_key = f'image-{idx+1:09d}', f'label-{idx+1:09d}'
            label = str(txn.get(label_key.encode()), 'utf-8')  # label
            label = re.sub('[^0-9a-zA-Z]+', '', label)
            label = label[:30]

            imgbuf = txn.get(image_key.encode())  # image
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            if self.rgb2gray:
                image = Image.open(buf).convert('L')
            else:
                image = Image.open(buf).convert('RGB')
            return image, label, image_key

    def __getitem__(self, idx):
        # self.data=[(img_path,text),...]
        image,label, idx = self.get(idx)

        if self.is_test==False and self.data_aug:
            image = self.augment_tfs(image)
        if self.keep_aspect_ratio:
            h, w = image.height, image.width
            ratio = w / h
            image = image.resize(
                (min(max(int(self.height * ratio), self.height), self.max_width), self.height),
                Image.ANTIALIAS
            )
        else:
            image = image.resize((self.width, self.height), Image.ANTIALIAS)
        image = np.array(image)
        if self.rgb2gray:
            image = np.expand_dims(image, -1)
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32) / 128. - 1.

        text = label
        if self.is_lower:
            text = [self.word2idx.get(ch.lower(), 1) for ch in text]
        else:
            text = [self.word2idx.get(ch, 1) for ch in text]
        text.insert(0, 2)
        text.append(3)
        target = np.array(text)
        if self.is_test:
            return image, label, idx
        return image, target

class MyConcatDataset(ConcatDataset):
    def __getattr__(self, k):
        return getattr(self.datasets[0], k)

def collate_fn(insts):
    # padding for normal size
    try:
        src_insts, tgt_insts = list(zip(*insts))
        # src_insts = torch.stack(src_insts, dim=0)
        src_insts = src_pad(src_insts)
        tgt_insts = tgt_pad(tgt_insts)
    except:
        return None
    return src_insts, tgt_insts


def collate_fn_test(insts):
    try:
        src_insts, gt_insts, name_insts = list(zip(*insts))
        # src_insts = torch.stack(src_insts, dim=0)
        src_insts = src_pad(src_insts)
    except:
        return None
    return src_insts, gt_insts, name_insts


def src_pad(insts):
    max_w = max(inst.shape[-1] for inst in insts)
    insts_ = []
    for inst in insts:
        d = max_w - inst.shape[-1]
        inst = np.pad(inst, ((0, 0), (0, 0), (0, d)), 'constant')
        insts_.append(inst)
    insts = torch.tensor(insts_).to(torch.float32)
    return insts


def tgt_pad(insts):
    # pad blank for size_len consist
    max_len = max(len(inst) for inst in insts)
    insts_ = []
    for inst in insts:
        d = max_len - inst.shape[0]
        inst = np.pad(inst, (0, d), 'constant')
        insts_.append(inst)
    batch_seq = torch.LongTensor(insts_)
    return batch_seq

def dataset_bag(ds_type,cfg,paths,word2idx, idx2word,is_train):
    if is_train == 2:
        is_test = True
    else:
        is_test = False
    datasets = [ds_type(
            image_dir=None,
            gt_file=p,
            word2idx=word2idx,
            idx2word=idx2word,
            size=(cfg.width, cfg.height),
            max_width=cfg.max_width,
            rgb2gray=cfg.rgb2gray,
            keep_aspect_ratio=cfg.keep_aspect_ratio,
            is_lower=cfg.is_lower,
            data_aug=cfg.data_aug,
            is_test=is_test) for p in paths]
    # if is_train ==1: return datasets
    if len(datasets) > 1: return MyConcatDataset(datasets)
    else: return datasets[0]

def make_data_loader(cfg, is_train=True,val_gt_file=None):
    vocab = cfg.dst_vocab
    vocab_size = cfg.dst_vocab_size
    word2idx, idx2word = load_vocab(vocab, vocab_size)
    tra_val_test = 0
    if is_train==False:
        tra_val_test = 1
    # is_train==false 表示为验证集
    # tra_val_test:0 train, 1 val, 2 test
    if is_train == False:
        dataset = dataset_bag(LMDBDataset,cfg,
                    [val_gt_file],
                    word2idx,idx2word,is_train = tra_val_test)
        # dataset = TXTDataset(
        #     image_dir=cfg.train.image_dir if is_train else cfg.val.image_dir,
        #     gt_file=cfg.train.gt_file if is_train else cfg.val.gt_file,
        #     word2idx=word2idx,
        #     idx2word=idx2word,
        #     size=(cfg.width, cfg.height),
        #     max_width=cfg.max_width,
        #     rgb2gray=cfg.rgb2gray,
        #     keep_aspect_ratio=cfg.keep_aspect_ratio,
        #     is_lower=cfg.is_lower,
        #     data_aug=cfg.data_aug,
        # )
    else:
        dataset = dataset_bag(LMDBDataset, cfg,
                    cfg.train.gt_file,
                    word2idx, idx2word,is_train=tra_val_test)
        # dataset = LMDBDataset(
        #     image_dir=cfg.train.image_dir if is_train else cfg.val.image_dir,
        #     gt_file=cfg.train.gt_file if is_train else cfg.val.gt_file,
        #     word2idx=word2idx,
        #     idx2word=idx2word,
        #     size=(cfg.width, cfg.height),
        #     max_width=cfg.max_width,
        #     rgb2gray=cfg.rgb2gray,
        #     keep_aspect_ratio=cfg.keep_aspect_ratio,
        #     is_lower=cfg.is_lower,
        #     data_aug=cfg.data_aug,
        # )
    if cfg.train_method=='dist':
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=cfg.train.batch_size if is_train else cfg.val.batch_size,
            num_workers=cfg.train.num_worker if is_train else cfg.val.num_worker,
            pin_memory=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )
    else:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=cfg.train.batch_size if is_train else cfg.val.batch_size,
            shuffle=True if is_train else False,
            num_workers=cfg.train.num_worker if is_train else cfg.val.num_worker,
            pin_memory=True,
            collate_fn=collate_fn,
        )
    return dataloader

def make_lmdb_data_loader_test(cfg, data_name, gt_file='test_gt.txt'):
    vocab = cfg.dst_vocab
    vocab_size = cfg.dst_vocab_size
    word2idx, idx2word = load_vocab(vocab, vocab_size)

    # image_dir = os.path.join(cfg.test.image_dir, data_name, 'test_image')
    # gt_file = os.path.join(cfg.test.image_dir, data_name, gt_file)
    dataset = dataset_bag(LMDBDataset, cfg,
                          paths = data_name,
                          word2idx=word2idx, idx2word=idx2word, is_train=2)
    # dataset=LMDBDataset(
    #     image_dir=image_dir,
    #     gt_file=gt_file,
    #     word2idx=word2idx,
    #     idx2word=idx2word,
    #     size=(cfg.width, cfg.height),
    #     is_test=True,
    #     max_width=cfg.max_width,
    #     rgb2gray=cfg.rgb2gray,
    #     keep_aspect_ratio=cfg.keep_aspect_ratio,
    #     is_lower=cfg.is_lower,
    # )
    if cfg.train_method=='dist':
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.test.batch_size,
        shuffle=False,
        num_workers=cfg.test.num_worker,
        pin_memory=True,
        collate_fn=collate_fn_test,
        sampler=train_sampler if cfg.train_method=='dist' else None,
    )
    return dataloader

def make_data_loader_test(cfg, data_name, gt_file='test_gt.txt'):
    vocab = cfg.dst_vocab
    vocab_size = cfg.dst_vocab_size
    word2idx, idx2word = load_vocab(vocab, vocab_size)
    image_dir = data_name
    gt_file = gt_file
    # image_dir = os.path.join(cfg.test.image_dir, data_name, 'test_image')
    # gt_file = os.path.join(cfg.test.image_dir, data_name, gt_file)
    dataset = TXTDataset(
    # dataset=LMDBDataset(
        image_dir=image_dir,
        gt_file=gt_file,
        word2idx=word2idx,
        idx2word=idx2word,
        size=(cfg.width, cfg.height),
        is_test=True,
        max_width=cfg.max_width,
        rgb2gray=cfg.rgb2gray,
        keep_aspect_ratio=cfg.keep_aspect_ratio,
        is_lower=cfg.is_lower,
    )
    if cfg.train_method=='dist':
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.test.batch_size,
        shuffle=False,
        num_workers=cfg.test.num_worker,
        pin_memory=True,
        collate_fn=collate_fn_test,
        sampler=train_sampler if cfg.train_method=='dist' else None,
    )
    return dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CDistNet')
    parser.add_argument('--config', type=str, help='train config file path')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    data_loader = make_data_loader(cfg, is_train=True)
    for idx, batch in enumerate(data_loader):
        print(batch[0].shape)
