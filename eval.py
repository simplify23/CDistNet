import os
import argparse
import codecs
import subprocess
import csv
import glob
# os.environ['CUDA_VISIBLE_DEVICES']="6"
from tqdm import tqdm
from mmcv import Config
import torch
import torch.nn as nn
import torch.distributed as dist

from cdistnet.data.data import make_data_loader_test, make_lmdb_data_loader_test
# from cdistnet.hdf5loader import make_data_loader
from cdistnet.model.translator import Translator
from cdistnet.utils.submit_with_lexicon import ic03_lex, iiit5k_lex, svt_lex, svt_p_lex
from cdistnet.model.model import build_CDistNet


def parse_args():
    parser = argparse.ArgumentParser(description='Train CDistNet')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--local_rank', default=-1, type=int,help='node rank for distributed training')
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


def get_alphabet(dict_path):
    with open(dict_path, "r") as f:
        data = f.readlines()
        data = list(map(lambda x: x.strip(), data))
        data = data[4:]
    return data


def get_pred_gt_name(translator, idx2word, b_image, b_gt, b_name, num, dict_path, rotate,rgb2gray,is_test_gt=True):
    # rgb2gray=False
    gt_list, name_list, pred_list = [], [], []
    alphabet = get_alphabet(dict_path)  # not used
    if rotate:
        batch_hyp, batch_scores = translator.translate_batch(
            images=b_image.view(-1, b_image.shape[-2], b_image.shape[-1]).unsqueeze(dim=1)
        )
        batch_scores = torch.cat(batch_scores, dim=0).view(-1, 3)
        _, idx = torch.max(batch_scores, 1)
        idx = torch.arange(0, idx.shape[0], dtype=torch.long) * 3 + idx.cpu()
        batch_hyp_ = []
        for id, v in enumerate(batch_hyp):
            if id in idx:
                batch_hyp_.append(v)
        batch_hyp = batch_hyp_
    else:
        if rgb2gray == False:
            batch_hyp, batch_scores = translator.translate_batch(images=b_image[:, :3, :, :])
        else:
            batch_hyp, batch_scores = translator.translate_batch(images=b_image[:, 0:1, :, :])
    for idx, seqs in enumerate(batch_hyp):
        for seq in seqs:
            seq = [x for x in seq if x != 3]
            pred = [idx2word[x] for x in seq]
            pred = ''.join(pred)
        flag = False
        if is_test_gt==False:
            num += 1
            pred_list.append('word_{}.png'.format(num) + ', "' + pred + '"\n')
            gt_list.append('word_{}.png'.format(num) + ', "' + b_gt[idx] + '"\n')
            name_list.append(b_name[idx] + '\n')
        else:
            num += 1
            pred_list.append('{}'.format(b_name[idx]) + ', "' + pred + '"\n')
            gt_list.append('{}'.format(b_name[idx]) + ', "' + b_gt[idx] + '"\n')
            name_list.append(b_name[idx] + '\n')
    return gt_list, name_list, pred_list, num


def write_to_file(file_name, datas):
    with open(file_name, "w") as f:
        f.writelines(datas)


def eval_and_save(script_path, gt_file, submit_file, python_path):
    cmd = "%s %s -g=%s -s=%s" % (python_path, script_path, gt_file, submit_file)
    print("cmd:{}".format(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    cmd_out = p.stdout.read().decode('utf-8')
    # Calculated!{"crwup": 0.9443155452436195, "tedupL": 0.9704312114989733, "tedL": 0.9704312114989733, "totalWords": 862, "crwN": 814.0, "crwupN": 814.0, "ted": 144.0, "tedup": 144.0, "detWords": 862, "crw": 0.9443155452436195}
    print("cmd_out: {}".format(cmd_out))
    crwup = cmd_out[cmd_out.index('{') + 1:cmd_out.index('}')].split(', ')[0]
    res = crwup.split(': ')[1]
    res = float(res) * 100
    res = '%.2f' % float(res)
    return str(res)


def start_eval(script_path, data_name, gt_file, pred_file, name_file, lexdir, python_path):
    submit_file_list = []
    res = [eval_and_save(script_path, gt_file, pred_file, python_path)]
    if data_name == 'icdar2003':
        submit_file_list.append(ic03_lex(os.path.join(lexdir, data_name), '50', gt_file, pred_file, name_file))
        submit_file_list.append(ic03_lex(os.path.join(lexdir, data_name), 'full', gt_file, pred_file, name_file))
    elif data_name == 'svt':
        submit_file_list.append(svt_lex(os.path.join(lexdir, data_name), '50', gt_file, pred_file, name_file))
    elif data_name == 'svt-p':
        submit_file_list.append(svt_p_lex(os.path.join(lexdir, data_name), '50', gt_file, pred_file, name_file))
    elif data_name == 'iiit5k':
        submit_file_list.append(iiit5k_lex(os.path.join(lexdir, data_name), '50', gt_file, pred_file, name_file))
        submit_file_list.append(iiit5k_lex(os.path.join(lexdir, data_name), '1k', gt_file, pred_file, name_file))
    for submit_file in submit_file_list:
        res.append(eval_and_save(script_path, gt_file, submit_file, python_path))
    return res

def start_eval_simple(submit_list,gt_list,name_list):
    i = 0
    total = 0
    num = len(gt_list)
    err_list = []
    for pred in submit_list:
        if pred.lower() == gt_list[i].lower():
            total+=1
        # else:
            # err_list.append("{} image is diff:{} ---- {}\n".format(name_list[i],pred,gt_list[i].lower()))
        i +=1
#     print(err_list)
#     with open(err_dir, "w") as f:
#         f.writelines(err_list)
    return total / num *100.0

def eval(cfg, args,model_path):
    # init dist_train
    if cfg.train_method=='dist':
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)

    model = build_CDistNet(cfg)
    model.load_state_dict(torch.load(model_path))
    if cfg.train_method=='dist':
        model.cuda(args.local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    else:
        device = torch.device(cfg.test.device)
        model.to(device)
    model.eval()

    translator = Translator(cfg, model=model)
    word2idx, idx2word = load_vocab(cfg.dst_vocab, cfg.dst_vocab_size)
    lexdir = cfg.test.image_dir
    result_line = []
    for i,data_name in enumerate(cfg.test.test_list):
        print("dataset name: {}".format(data_name))
        if cfg.test.is_test_gt ==True:
            test_dataloader = make_data_loader_test(cfg, lexdir[i], gt_file=data_name)
        else:
            test_dataloader = make_lmdb_data_loader_test(cfg, [data_name])

        gt_list, name_list, pred_list = [], [], []
        num = 0

        #start eval
        for iteration, batch in enumerate(tqdm(test_dataloader)):
            b_image, b_gt, b_name = batch[0], batch[1], batch[2]
            gt_list_, name_list_, pred_list_, num = get_pred_gt_name(
                translator, idx2word, b_image, b_gt, b_name, num, cfg.dst_vocab, cfg.test.rotate,cfg.rgb2gray,cfg.test.is_test_gt
            )
            gt_list += gt_list_
            name_list += name_list_
            pred_list += pred_list_
        # print("gt:{} \n pred:{}".format(gt_list,pred_list))
        gt_file = os.path.join(cfg.test.model_dir, 'gt.txt')
        pred_file = os.path.join(cfg.test.model_dir, 'submit.txt')
        name_file = os.path.join(cfg.test.model_dir, 'name.txt')
        write_to_file(gt_file, gt_list)
        write_to_file(pred_file, pred_list)
        write_to_file(name_file, name_list)
        res_simple = start_eval_simple(pred_list,gt_list,name_list)
        print("res_simple_acc:{}".format(res_simple))
        result_line += res_simple
        # if cfg.test.is_test_gt == False:
        #     res = start_eval(cfg.test.script_path, data_name, gt_file, pred_file, name_file, lexdir, cfg.test.python_path)
        #     result_line += res
        # print("result_line:{}".format(result_line))
    result_line.insert(0, model_path.split('/')[-1])
    print(os.path.join(cfg.test.model_dir, 'result.csv'))
    with open(os.path.join(cfg.test.model_dir, 'result.csv'), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(result_line)


def average(model, models):
    """Average models into model"""
    # with torch.no_grad():
    #     for ps in zip(*[m.parameters() for m in [model] + models]):
    #         ps[0].copy_(torch.sum(torch.stack(ps[1:]), dim=0) / len(ps[1:]))

    with torch.no_grad():
        for key in model.state_dict().keys():
            v = []
            for m in models:
                v.append(m.state_dict()[key])
            v = torch.sum(torch.stack(v), dim=0) / len(v)
            model.state_dict()[key].copy_(v)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    headers = cfg.test.test_list
    result_path = os.path.join(cfg.test.model_dir, 'result.csv')
    if not os.path.exists(result_path):
        with open(result_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    if cfg.test.best_acc_test:
        path2 = glob.glob(cfg.test.model_dir + '/epoch9_*.pth')
        path = glob.glob(cfg.test.model_dir + '/*_best_acc.pth')
        for model_path in path2:
            print("model: {}".format(model_path))
            # eval(cfg, args,os.path.join(cfg.test.model_dir, model_path))
            eval(cfg, args, model_path)
        for model_path in path:
            print("model: {}".format(model_path))
            # eval(cfg, args,os.path.join(cfg.test.model_dir, model_path))
            eval(cfg, args, model_path)
        # return

    # eval all
    if cfg.test.eval_all:
        paths = glob.glob(cfg.test.model_dir + "/*.pth")
        for model_path in paths:
            print("model: {}".format(model_path))
            # eval(cfg, args,os.path.join(cfg.test.model_dir, model_path))
            eval(cfg, args, model_path)
            return
    else:
        model_path_patten = cfg.test.model_dir + '/model_epoch_{}.pth'
        s, e = cfg.test.s_epoch, cfg.test.e_epoch
        if e < s:
            s, e = e, s
        if s != -1:
            for i in range(s, e + 1):
                model_path = model_path_patten.format(i)
                print("model: {}".format(model_path))
                eval(cfg, args,model_path)

        # model average
        avg_s, avg_e = cfg.test.avg_s, cfg.test.avg_e
        if avg_e < avg_s:
            avg_s, avg_e = avg_e, avg_s
        if avg_s == -1:
            return
        models = []
        if cfg.test.avg_all:
            for i in range(avg_s, avg_e + 1):
                model_paths = glob.glob(cfg.test.model_dir + '/model_epoch_{}*.pth'.format(i))
                for model_path in model_paths:
                    print("model: {}".format(model_path))
                    model = build_CDistNet(cfg)
                    model.load_state_dict(torch.load(model_path))
                    models.append(model)
        else:
            for i in range(avg_s, avg_e + 1):
                model_path = model_path_patten.format(i)
                print("model: {}".format(model_path))
                model = build_CDistNet(cfg)
                model.load_state_dict(torch.load(model_path))
                models.append(model)
        model = build_CDistNet(cfg)
        # model = models[0]
        average(model, models)
        if cfg.test.avg_all:
            model_path = os.path.join(cfg.test.model_dir, 'model_epoch_avg({}-{}-all).pth'.format(avg_s, avg_e))
        else:
            model_path = os.path.join(cfg.test.model_dir, 'model_epoch_avg({}-{}).pth'.format(avg_s, avg_e))
        torch.save(model.state_dict(), model_path)
        eval(cfg, args,model_path)


if __name__ == '__main__':
    main()
