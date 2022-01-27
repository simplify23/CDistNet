import argparse
import codecs

import cv2
import torch
from PIL import Image
from tqdm import tqdm
from mmcv import Config
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# from cdistnet.hdf5loader import make_data_loader
from cdistnet.model.translator import Translator
from cdistnet.model.model import build_CDistNet
# from cdistnet.data.data import make_data_loader


# test

def parse_args():
    parser = argparse.ArgumentParser(description='Train CDistNet')
    parser.add_argument('--i_path', type=str, default='1.jpg',
                        help='Input image path')
    parser.add_argument('--model_path', type=str, default='models/new_baseline_dssnetv3_3_32*128_tps_resnet45_epoch_6/epoch9_best_acc.pth',
                        help='Input model path')
    parser.add_argument('--config', type=str, default='configs/CDistNet_config.py',
                        help='train config file path')
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--test_one', default=True,
                        help='test one image')
    parser.add_argument('--use_origin', default=True,
                        help='use_origin_process')
    args = parser.parse_args()
    return args


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


def preprocess_image(image_path):
    img = cv2.imread(image_path, 1)
    assert img is not None
    img = np.float32(img)
    # # Opencv loads as BGR:
    img = img[:, :, ::-1]
    grayscale = transforms.Grayscale(num_output_channels=1)
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        # normalize,
        transforms.ToPILImage(),
        grayscale,
        transforms.ToTensor(),
    ])
    return preprocessing(img.copy()).unsqueeze(0)


def origin_process_img(cfg, image_path):
    # self.data=[(img_path,text),...]
    if cfg.rgb2gray:
        image = Image.open(image_path).convert('L')
    else:
        image = Image.open(image_path).convert('RGB')
    assert image is not None
    image = image.resize((cfg.width, cfg.height), Image.ANTIALIAS)
    image = np.array(image)
    if cfg.rgb2gray:
        image = np.expand_dims(image, -1)
        image = np.expand_dims(image, -1)
    print(image.shape)
    image = np.expand_dims(image, -1)
    image = image.transpose((2, 3, 0, 1))
    image = image.astype(np.float32) / 128. - 1.
    image = torch.from_numpy(image)
    # text = self.data[idx][1]
    # text = [self.word2idx.get(ch, 1) for ch in text]
    # text.insert(0, 2)
    # text.append(3)
    # target = np.array(text)
    return image


def test(cfg):
    model = build_CDistNet(cfg)
    model.load_state_dict(torch.load(
        '/media/zs/zs/zs/code/NRTR/models/baseline_hdf5_100_32_two_local_MultiHeadAttention/model_epoch_avg.pth'))
    device = torch.device(cfg.test.device)
    model.to(device)
    model.eval()
    cfg.n_best = 5
    # vision more res
    translator = Translator(cfg, model)
    val_dataloader = make_data_loader(cfg, is_train=False)
    word2idx, idx2word = load_vocab('datasets/en_vocab', 40)
    cnt = 1
    with open('pred.txt', 'w') as f:
        for batch in tqdm(val_dataloader):
            all_hyp, all_scores = translator.translate_batch(batch[0])
            for idx_seqs in all_hyp:
                for idx_seq in idx_seqs:
                    idx_seq = [x for x in idx_seq if x != 3]
                    pred_line = '{}.png, "'.format(cnt) + ''.join([idx2word[idx] for idx in idx_seq]) + '"'
                    f.write(pred_line + '\n')
                    cnt += 1

def get_parameter_number(net):
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return 'Trainable: {} M'.format(trainable_num/1000000)

def test_one(cfg, args):
    # model_path = 'models/baseline_20_epoch_wh_44/model_epoch_10.pth'
    # prepare model
    model = build_CDistNet(cfg)
    en = get_parameter_number(model.transformer.encoder)
    de = get_parameter_number(model.transformer.decoder)
    print('encoder:{}\ndecoder:{}\n'.format(en,de))
    model_path = 'models/new_baseline_dssnetv3_3_32*128_tps_resnet45_epoch_6/epoch9_best_acc.pth'
    model.load_state_dict(torch.load(model_path))
    device = torch.device(cfg.test.device)
    model.to(device)
    model.eval()
    translator = Translator(cfg, model)
    word2idx, idx2word = load_vocab(cfg.dst_vocab, cfg.dst_vocab_size)

    # if args['use_origin'] is True:
    img = origin_process_img(cfg, 'test/1.jpg')
    # else:
    #     img = preprocess_image(args['img_path'])

    cnt = 0
    res = []
    all_hyp, all_scores = translator.translate_batch(img)
    # print(all_hyp, all_scores)
    for idx_seqs in all_hyp:
        for idx_seq in idx_seqs:
            idx_seq = [x for x in idx_seq if x != 3]
            pred_line = 'Results{}:"'.format(cnt) + ''.join([idx2word[idx] for idx in idx_seq]) + '"'
            res.append('Vocab Prob:{}\nTotal Score:{}\n{}\n\n'\
                .format(all_hyp[0][cnt],all_scores[0][cnt],pred_line))
            cnt = cnt + 1
    print(res)
    return res

def test_demo(args):
    print(args['config_path'])
    print(type(args['config_path']))
    cfg = Config.fromfile(args['config_path'])
    return test_one(cfg, args)

def main():
    args = parse_args()
    print(args.config)
    print(type(args.config))
    cfg = Config.fromfile(args.config)
    if args.test_one is True:
        test_one(cfg, args)
    else:
        test(cfg)


if __name__ == '__main__':
    # test_demo()
    main()