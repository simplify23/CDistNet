import os
import argparse
import glob
import numpy as np
import h5py
import cv2
import codecs
from tqdm import tqdm
from PIL import Image, ImageFile

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


def get_train_addrs(hdf5_path, img_path, gt_path, en_vocab_path, max_text_len, dict_size):
    addrs, labels = [], []
    word2idx, idx2word = load_vocab(en_vocab_path, dict_size)

    with open(gt_path, 'r', encoding='UTF-8-sig') as f:
        all = f.readlines()
        max_len = -1
        for each in tqdm(all):
            each = each.strip().split(' ')
            path = os.path.join(img_path, each[0])
            text = " ".join(each[1:])
            # text = [word2idx.get(ch.lower(), 1) for ch in text]
            text = [word2idx.get(ch, 1) for ch in text]
            text.insert(0, 2)
            text.append(3)
            max_len = max(max_len, len(text))
            text = np.array(text)
            text = np.pad(text, (0, max_text_len - text.size), 'constant')
            labels.append(text)
            addrs.append(path)
        # print(max_len)

    c = list(zip(addrs, labels))
    addrs, labels = zip(*c)
    train_addrs = addrs
    train_labels = labels
    return hdf5_path, train_addrs, train_labels


def create_hdf5_file(hdf5_path, train_addrs, train_labels, keep_aspect_ratio=False, height=32, max_width=180, max_text_len=35):
    train_shape = (len(train_addrs), 1, height, max_width if keep_aspect_ratio else 100)
    hdf5_file = h5py.File(hdf5_path, mode='w')
    hdf5_file.create_dataset("image", train_shape, np.float32)
    hdf5_file.create_dataset("label", (len(train_addrs), max_text_len), np.int)
    hdf5_file["label"][...] = train_labels
    return hdf5_file, train_shape


def load_and_save_image(train_addrs, hdf5_file, keep_aspect_ratio=False, max_width=180, height=32):
    for i in tqdm(range(len(train_addrs))):
        addr = train_addrs[i]
        try:
            if keep_aspect_ratio:
                img = Image.open(addr).convert('L')
                # img = cv2.imread(addr, cv2.IMREAD_GRAYSCALE)
                h, w = img.height, img.width
                # h, w = img.shape
                r = w * 1.0 / h
                r_h, r_w = height, min(max(height * r, height), max_width)

                img = img.resize((int(r_w), r_h), Image.ANTIALIAS)
                # img = cv2.resize(img, (int(r_w), r_h), interpolation=cv2.INTER_CUBIC)
                img = np.array(img, dtype=np.uint8)
                img = np.expand_dims(img, -1)
                img = img.transpose((2, 0, 1))
                img = img.astype(np.float32) / 128. - 1.
                d = max_width - img.shape[-1]
                img = np.pad(img, ((0, 0), (0, 0), (0, d)), 'constant')
            else:
                img = Image.open(addr).convert('L').resize((100, 32), Image.ANTIALIAS)
                img = np.array(img, dtype=np.uint8)
                # img = cv2.imread(addr, cv2.IMREAD_GRAYSCALE)
                # img = cv2.resize(img, (100, 32), interpolation=cv2.INTER_CUBIC)
                img = np.expand_dims(img, -1)
                img = img.transpose((2, 0, 1))
                img = img.astype(np.float32) / 128. - 1.
        except:
            print(addr)
            img = np.zeros((1, height, max_width if keep_aspect_ratio else 100), dtype=np.float32)
        hdf5_file["image"][i, ...] = img[None]
    hdf5_file.close()


def main():
    parser = argparse.ArgumentParser(description='Train NRTR')
    parser.add_argument('--hdf5_path', type=str, default='')
    parser.add_argument('--img_path', type=str, default='')
    parser.add_argument('--gt_path', type=str, default='')
    parser.add_argument('--keep_aspect_ratio', action='store_true')
    parser.add_argument('--max_width', type=int, default=180)
    parser.add_argument('--height', type=int, default=32)
    parser.add_argument('--en_vocab_path', type=str, default='')
    parser.add_argument('--max_text_len', type=int, default=35)
    parser.add_argument('--dict_size', type=int, default=40)
    args = parser.parse_args()
    # hdf5_path = '../datasets/train_two_keep.hdf5'
    # img_path = '/home/zhengsheng/dataset/reg'
    # gt_path = '/home/zhengsheng/dataset/reg/annotation_train_clean.txt'
    # hdf5_path = '../datasets/train_two_keep_aspect_ratio.hdf5'
    # img_path = '../datasets/image'
    # gt_path = '../datasets/gt/new_gt.txt'
    # keep_aspect_ratio = True
    # max_width = 180
    # height = 32
    # en_vocab_path = '/home/zs/zs/code/NRTR/datasets/en_vocab'

    hdf5_path = args.hdf5_path
    img_path = args.img_path
    gt_path = args.gt_path
    keep_aspect_ratio = args.keep_aspect_ratio
    height = args.height
    max_width = args.max_width
    en_vocab_path = args.en_vocab_path
    max_text_len = args.max_text_len
    dict_size = args.dict_size
    print("hdf5_path: ", hdf5_path)
    print("img_path: ", img_path)
    print("gt_path: ", gt_path)
    print("keep_aspect_ratio: ", keep_aspect_ratio)
    print("height: ", height)
    print("max_width:", max_width)
    print("en_vocab_path: ", en_vocab_path)
    print("max_text_len: ", max_text_len)
    print("dict_size: ", dict_size)
    hdf5_path, train_addrs, train_labels = get_train_addrs(hdf5_path, img_path, gt_path, en_vocab_path, max_text_len, dict_size)
    hdf5_file, train_shape = create_hdf5_file(hdf5_path, train_addrs, train_labels, keep_aspect_ratio, height, max_width, max_text_len)
    load_and_save_image(train_addrs, hdf5_file, keep_aspect_ratio, max_width, height)


# def test():
#     hdf5_file = h5py.File("/home/psdz/datasets/train_three_and_chinese.hdf5", "r")
#     print(len(hdf5_file['label']))
#     for i in range(len(hdf5_file['label'])):
#         label = hdf5_file['label'][i]
#         print(label)
#         if i == 100:
#             break
#         # Image.fromarray(hdf5_file['image'][i].astype('uint8')).save('./tmp.jpg')
#         # print(hdf5_file['flag'][i])
#         # break


if __name__ == '__main__':
    main()