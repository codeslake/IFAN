import torch
import collections

from utils import *
from models.utils import *
from data_loader.FastDataLoader import FastDataLoader

class baseModel():
    def __init__(self, config):
        self.config = config
        self.is_train = config.is_train

        self.network = None
        self.results = collections.OrderedDict()

        self.schedulers = []
        self.optimizers = []

    def _create_dataloader(self, dataset, sampler, is_train, wif = None):
        drop_last = False
        if is_train:
            shuffle = False if self.config.dist else True

            data_loader = FastDataLoader(
                            dataset,
                            batch_size=self.config.batch_size if is_train else self.config.batch_size_test,
                            shuffle=shuffle,
                            num_workers=self.config.thread_num,
                            sampler=sampler,
                            drop_last=drop_last,
                            worker_init_fn = wif,
                            pin_memory=True)

        else:
            shuffle = False
            data_loader = torch.utils.data.DataLoader(
                            dataset,
                            batch_size=self.config.batch_size if is_train else self.config.batch_size_test,
                            shuffle=shuffle,
                            num_workers=self.config.thread_num,
                            sampler=sampler,
                            drop_last=drop_last,
                            worker_init_fn = wif,
                            pin_memory=False)

        return data_loader

    def _set_lr(self, lr_groups_l):
        """Set learning rate for warmup
        lr_groups_l: list for lr_groups. each for a optimizer"""
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        """Get the initial lr, which is set by the scheduler"""
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def _update_learning_rate(self, cur_itr, warmup_itr=-1):
        for scheduler in self.schedulers:
            scheduler.step()
        # set up warm-up learning rate
        if cur_itr < warmup_itr:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_itr * cur_itr for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

        for optimizer in self.optimizers:
            lr = [v['lr'] for v in optimizer.param_groups]

        return lr 

    def _update(self, errs, warmup_itr = -1):
        self.optimizer.zero_grad()
        errs['total'].backward()
        self.optimizer.step()
        lr = self._update_learning_rate(self.itr_global['train'], warmup_itr)
        
        return lr

    def _set_visuals(self, inputs, outs, errs):
        self.visuals = collections.OrderedDict()

    def get_network(self):
        return self.network

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()

    def print(self):
        print(self.network)

    def get_training_state(self, epoch):
        """Save training state during training, which will be used for resuming"""
        state = {'epoch': epoch, 'itr': self.itr_global, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())

        return state

    def resume_training(self, resume_state):
        """Resume the optimizers and schedulers for training"""
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        self.itr_global = resume_state['itr']

        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'

        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
            #resetting learning rate while resuming (learning rate of MultiStepLR_Restart and CosineAnneaelingLR_Restart will be reset)
            for g in self.optimizers[i].param_groups:
                g['lr'] = self.config.lr_init
                g['initial_lr'] = self.config.lr_init

        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)
            if 'MultiStepLR_Restart' in type(self.schedulers[i]).__name__:
                self.schedulers[i].reset_param(self.config.resetarts, self.config.restart_weights)
            elif 'CosineAnnealingLR_Restart' in type(self.schedulers[i]).__name__:
                self.schedulers[i].reset_param(self.config.T_period, self.config.restarts, self.config.restart_weights)
            elif 'LR_decay' in type(self.schedulers[i]).__name__:
                self.schedulers[i].reset_param(self.config.decay_period, self.config.decay_rate)
            elif 'LR_decay_progressive' in type(self.schedulers[i]).__name__:
                self.schedulers[i].reset_param(self.config.decay_period, self.config.decay_rate)
