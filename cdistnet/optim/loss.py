import torch
import torch.nn.functional as F
from torch import distributed

def reduce_tensor(tensor):
    rt = tensor.clone()
    distributed.all_reduce(rt, op=distributed.reduce_op.SUM)
    rt /= distributed.get_world_size()#总进程数
    return rt

def cal_performance(pred, tgt, local_rank,smoothing=True):
    # pred(b*tgt_len,vacab_size)
    # tgt(b,max_len)
    loss = cal_loss(pred, tgt, local_rank, smoothing)
    pred = pred.max(1)[1]
    tgt = tgt.contiguous().view(-1)
    non_pad_mask = tgt.ne(0)
    n_correct = pred.eq(tgt)

    # loss = reduce_tensor(loss.data)
    # n_correct = reduce_tensor(n_correct)

    n_correct = n_correct.masked_select(non_pad_mask).sum().item()
    return loss, n_correct


def cal_loss(pred, tgt, local_rank, smoothing=True):
    tgt = tgt.contiguous().view(-1)
    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, tgt.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = tgt.ne(0)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).mean()
    else:
        loss = F.cross_entropy(pred, tgt, ignore_index=0, reduction='mean')
    return loss