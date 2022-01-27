import numpy as np


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps, n_current_steps=0):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = n_current_steps
        self.init_lr = np.power(d_model, -0.5)
        # curr_epoch = step + 1 (4 -- 5th)
        self.step2 = 7
        self.step2_lr = 0.00001

    def step_and_update_lr(self,epoch = 0):
        "Step with the inner optimizer"
        self._update_learning_rate(epoch)
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self,epoch):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()
        if epoch >= self.step2:
            lr = self.step2_lr

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # lmbda = lambda epoch: 0.9**(epoch // 300) if epoch < 13200 else 10**(-2)
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)

    # if self.last_epoch < self.warmup_steps:
    #     return (self.end_lr - self.start_lr) * float(
    #         self.last_epoch) / float(self.warmup_steps) + self.start_lr

class WarmupOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps, n_current_steps=0,current_epoch=0):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.current_epoch =  current_epoch
        self.n_current_steps = n_current_steps
        self.start_lr = 0.0
        # self.init_lr = 0.001
        self.step_lr = [0.001,0.0001,0.00001]
        self.step = [4,6]  #curr_epoch = step + 1 (4 -- 5th)

    def step_and_update_lr(self,epoch):
        "Step with the inner optimizer"
        self._update_learning_rate(epoch)
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self,epoch):
        if epoch <= self.step[0]:
            return self.step_lr[0]
        if epoch <= self.step[1]:
            return self.step_lr[1]
        return self.step_lr[2]

    def _update_learning_rate(self,epoch):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        if self.n_current_steps < self.n_warmup_steps:
            lr = (self.step_lr[0] - self.start_lr) * float(
                self.n_current_steps) / float(self.n_warmup_steps) + self.start_lr
        else:
            lr = self._get_lr_scale(epoch)

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr