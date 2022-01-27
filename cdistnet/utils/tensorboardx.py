import time
from datetime import datetime
import torch
from tensorboardX import SummaryWriter

class TensorboardLogger(object):
    def __init__(self, log_dir, start_iter=0):
        self.iteration = start_iter
        self.writer = self._get_tensorboard_writer(log_dir)

    @staticmethod
    def _get_tensorboard_writer(log_dir):
        timestamp = datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H:%M')
        tb_logger = SummaryWriter('{}/{}'.format(log_dir, timestamp))
        return tb_logger

    def add_scalar(self, ** kwargs):
        if self.writer:
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                assert isinstance(v, (float, int))
                self.writer.add_scalar(k, v, self.iteration)

    def add_graph(self,model,data=(torch.randn(4,32),torch.randn(4,1,128,32))):
        # tgt(b,max_len)
        # image(b,c,h,w)
        self.writer.add_graph(model,input_to_model=data)

    def add_histogram(self,k,v):
        self.writer.add_histogram(k,v,self.iteration)

    def update_iter(self, iteration):
        self.iteration = iteration