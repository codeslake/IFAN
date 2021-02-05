import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import json

import time
import numpy
import os
import sys
import collections
import numpy as np
import gc
import math
import random
# import wandb

from models import create_model
from utils import *
from ckpt_manager import CKPT_Manager

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#torch.backends.cudnn.enabled = False
#torch.backends.cudnn.benchmark = False

class Trainer():
    def __init__(self, config, rank = -1):
        self.rank = rank
        if config.dist:
            self.pg = dist.new_group(range(dist.get_world_size()))

        self.config = config
        if self.rank <= 0: self.summary = SummaryWriter(config.LOG_DIR.log_scalar)

        ## model
        self.model = create_model(config)
        # if self.rank <= 0 and config.is_verbose:
        #     self.model.print()

        ## checkpoint manager
        self.ckpt_manager = CKPT_Manager(config.LOG_DIR.ckpt, config.mode, config.max_ckpt_num, is_descending = True)

        ## training vars
        self.states = ['train', 'valid']
        # self.states = ['valid', 'train']
        self.max_epoch = int(math.ceil(config.total_itr / self.model.get_itr_per_epoch()))
        self.epoch_range = np.arange(1, self.max_epoch + 1)
        self.itr = 0
        self.err_epoch = {'train': {}, 'valid': {}}
        self.norm = torch.tensor(0).to(torch.device('cuda'))
        self.lr = 0

        if self.config.resume is not None:
            if self.rank <= 0: print(toGreen('Resume Trianing...'))
            if self.rank <= 0: print(toRed('\tResuming {}..'.format(self.config.resume)))
            resume_state = self.ckpt_manager.resume(self.model.get_network(), self.config.resume, self.rank)
            # if 'resume' in self.config.mode:
            #     self.max_epoch = 5666
            self.epoch_range = np.arange(resume_state['epoch'] + 1, self.max_epoch + 1)
            self.model.resume_training(resume_state)


    def train(self):
        torch.backends.cudnn.benchmark = True
        if self.rank <= 0 : print(toYellow('======== TRAINING START ========='))
        for epoch in self.epoch_range:
            if self.rank <= 0 and epoch == 1:
                if self.config.resume is None:
                    self.ckpt_manager.save(self.model.get_network(), self.model.get_training_state(0), 0, score = [1e-8, 1e8])
            is_log = epoch == 1 or epoch % self.config.write_ckpt_every_epoch == 0 or epoch > self.max_epoch - 10

            for state in self.states:
                epoch_time = time.time()
                self.itr = 0
                if state == 'train':
                    self.model.train()
                    self.iteration(epoch, state, is_log)
                elif is_log:
                    self.model.eval()
                    with torch.no_grad():
                        self.iteration(epoch, state, is_log)

                if state == 'valid' or state == 'train' : # add "or state == 'train" if you want to save train logs
                    if is_log:
                        if config.dist: dist.all_reduce(self.norm, op=dist.ReduceOp.SUM, group=self.pg, async_op=False)
                        for k, v in self.err_epoch[state].items():
                            if config.dist:  dist.all_reduce(self.err_epoch[state][k], op=dist.ReduceOp.SUM, group=self.pg, async_op=False)
                            self.err_epoch[state][k] = (self.err_epoch[state][k] / self.norm).item()

                            if self.rank <= 0:
                                self.summary.add_scalar('loss/epoch_{}_{}'.format(state, k), self.err_epoch[state][k], epoch)

                        if self.rank <= 0:
                            torch.cuda.synchronize()
                            print_logs(state.upper() + ' TOTAL', self.config.mode, epoch, self.max_epoch, epoch_time, errs = self.err_epoch[state], lr = self.lr)
                            if state == 'valid':
                                is_saved = False
                                while is_saved == False:
                                    try:
                                        if math.isnan(self.err_epoch['valid']['psnr']) == False:
                                            self.ckpt_manager.save(self.model.get_network(), self.model.get_training_state(epoch), epoch, score = [self.err_epoch['valid']['psnr'], self.err_epoch['valid']['LPIPS']])
                                        is_saved = True
                                    except Exception as ex:
                                        is_saved = False

                        self.err_epoch[state] = {}
                        if config.dist:
                            dist.barrier()

            if self.rank <= 0:
                if epoch % self.config.refresh_image_log_every_epoch['train'] == 0:
                    remove_file_end_with(self.config.LOG_DIR.sample, '*.jpg')
                    remove_file_end_with(self.config.LOG_DIR.sample, '*.png')
                if epoch % self.config.refresh_image_log_every_epoch['valid'] == 0:
                    remove_file_end_with(self.config.LOG_DIR.sample_val, '*.jpg')
                    remove_file_end_with(self.config.LOG_DIR.sample_val, '*.png')

            gc.collect()

    def iteration(self, epoch, state, is_log):


        is_train = True if state == 'train' else False
        data_loader = self.model.data_loader_train if is_train else self.model.data_loader_eval
        if config.dist:
            if is_train: self.model.sampler_train.set_epoch(epoch)


        self.norm = torch.tensor(0).to(torch.device('cuda'))
        for inputs in data_loader:
            lr = None
            itr_time = time.time()

            self.model.iteration(inputs, epoch, self.max_epoch, is_train)
            self.itr += 1

            if is_log:
                bs = inputs['gt'].size()[0]
                errs = self.model.results['errs']
                for k, v in errs.items():
                    v = bs * v
                    if self.itr == 1:
                        self.err_epoch[state][k] = v
                    else:
                        if k in self.err_epoch[state].keys():
                            self.err_epoch[state][k] += v
                        else:
                            self.err_epoch[state][k] = v
                self.norm = self.norm + bs 
                if self.rank <= 0:
                    ## saves image patches for logging
                    # inputs = self.model.results['inputs']
                    # outs = self.model.results['outs']
                    # sample_dir = self.config.LOG_DIR.sample if is_train else self.config.LOG_DIR.sample_val
                    # if self.itr == 1 or self.model.itr_global[state] % config.write_log_every_itr[state] == 0:
                        # try:
                        #     # for key, val in errs.items():
                        #     #     self.summary.add_scalar('loss/{}_{}'.format(key, state), val, self.model.itr_global[state])

                        #     # vutils.save_image(inputs['input'], '{}/{}_{}_1_input.jpg'.format(sample_dir, epoch, self.itr), nrow=3, padding = 0, normalize = False)
                        #     # # vutils.save_image((inputs['input'].cpu().numpy() * self.config.norm_val).astype(np.uint16), '{}/{}_{}_1_input.jpg'.format(sample_dir, epoch, self.itr), nrow=3, padding = 0, normalize = False)
                        #     # i = 2
                        #     # for key, val in outs.items():
                        #     #     if val.dim() == 5:
                        #     #         for j in range(val.size()[1]):
                        #     #             vutils.save_image(val[:, j, :, :, :], '{}/{}_{}_{}_out_{}_{}.jpg'.format(sample_dir, epoch, self.itr, i, key, j), nrow=3, padding = 0, normalize = False)
                        #     #     else:
                        #     #         vutils.save_image(val, '{}/{}_{}_{}_out_{}.jpg'.format(sample_dir, epoch, self.itr, i, key), nrow=3, padding = 0, normalize = False)
                        #     #     i += 1

                        #     # vutils.save_image(inputs['gt'], '{}/{}_{}_{}_gt.jpg'.format(sample_dir, epoch, self.itr, i), nrow=3, padding = 0, normalize = False)
                        # except Exception as ex:
                        #     if self.rank <= 0 :  print('saving error: ', ex)

                    self.lr = self.model.results['lr']
                    print_logs(state.upper(), self.config.mode, epoch, self.max_epoch, itr_time, self.itr, len(data_loader), errs = errs, lr = self.lr)

##########################################################
def init_dist(backend='nccl', **kwargs):
    """initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)

if __name__ == '__main__':
    project = 'Defocus_Deblurring'
    mode = 'IFAN'

    import importlib
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--is_train', action = 'store_true', default = False, help = 'whether to delete log')
    parser.add_argument('--config', type = str, default = None, help = 'config name')
    parser.add_argument('--mode', type = str, default = mode, help = 'mode name')
    parser.add_argument('--project', type = str, default = project, help = 'project name')
    args, _ = parser.parse_known_args()

    if args.is_train:
        config_lib = importlib.import_module('configs.{}'.format(args.config))
        config = config_lib.get_config(args.project, args.mode, args.config)
        config.is_train = True

        ## DEFAULT
        parser.add_argument('-trainer', '--trainer', type = str, default = config.mode, help = 'model name')
        parser.add_argument('-net', '--network', type = str, default = config.mode, help = 'network name')
        parser.add_argument('-r', '--resume', type = str, default = config.resume, help = 'name of state or ckpt (names are the same)')
        parser.add_argument('-dl', '--delete_log', action = 'store_true', default = False, help = 'whether to delete log')
        parser.add_argument('-lr', '--lr_init', type = float, default = config.lr_init, help = 'leraning rate')
        parser.add_argument('-b', '--batch_size', type = int, default = config.batch_size, help = 'number of batch')
        parser.add_argument('-th', '--thread_num', type = int, default = config.thread_num, help = 'number of thread')
        parser.add_argument('-dist', '--dist', action = 'store_true', default = config.dist, help = 'whether to distributed pytorch')
        parser.add_argument('-vs', '--is_verbose', action = 'store_true', default = False, help = 'whether to delete log')
        parser.add_argument("--local_rank", type=int)

        ## CUSTOM
        parser.add_argument('-wi', '--weights_init', type = float, default = config.wi, help = 'weights_init')
        parser.add_argument('-lmdb', '--lmdb', action = 'store_true', default = False, help = 'whether to use lmdb')
        parser.add_argument('-proc', '--proc', type = str, default = 'proc', help = 'dummy process name for killing')
        parser.add_argument('-gc', '--gc', type = float, default = config.gc, help = 'gradient clipping')

        args, _ = parser.parse_known_args()

        ## default
        config.trainer = args.trainer
        config.network = args.network

        config.resume = args.resume
        config.delete_log = False if config.resume is not None else args.delete_log
        config.lr_init = args.lr_init
        config.batch_size = args.batch_size
        config.thread_num = args.thread_num
        config.dist = args.dist
        config.is_verbose = args.is_verbose

        # CUSTOM
        config.wi = args.weights_init
        config.lmdb = args.lmdb
        config.gc = args.gc


        if config.dist:
            init_dist()
            rank = dist.get_rank()
        else:
            rank = -1

        if rank <= 0:
            handle_directory(config, config.delete_log)
            print(toGreen('Laoding Config...'))
            # config_lib.print_config(config)
            config_lib.log_config(config.LOG_DIR.config, config)
            print(toRed('\tProject : {}'.format(config.project)))
            print(toRed('\tMode : {}'.format(config.mode)))
            print(toRed('\tConfig: {}'.format(config.config)))
            print(toRed('\tNetwork: {}'.format(config.network)))
            print(toRed('\tTrainer: {}'.format(config.trainer)))

        if config.dist:
            dist.barrier()

        ## random seed
        seed = config.manual_seed
        if seed is None:
            seed = random.randint(1, 10000)
        if rank <= 0 and config.is_verbose: print('Random seed: {}'.format(seed))

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        trainer = Trainer(config, rank)
        trainer.train()

    else:
        from eval import *
        from configs.config import get_config, set_eval_path
        from easydict import EasyDict as edict
        print(toGreen('Laoding Config for evaluation'))
        if args.config is None:
            config = get_config(args.project, args.mode, None)
            with open('{}/config.txt'.format(config.LOG_DIR.config)) as json_file:
                json_data = json.load(json_file)
                # config_lib = importlib.import_module('configs.{}'.format(json_data['config']))
                config = edict(json_data)
                # print(config['config'])
        else:
            config_lib = importlib.import_module('configs.{}'.format(args.config))

        config = config_lib.get_config(args.project, args.mode, args.config)

        config.is_train = False
        ## EVAL
        parser.add_argument('-net', '--network', type = str, default = config.mode, help = 'network name')
        parser.add_argument('-ckpt_name', '--ckpt_name', type=str, default = None, help='ckpt name')
        parser.add_argument('-ckpt_abs_name', '--ckpt_abs_name', type=str, default = None, help='ckpt abs name')
        parser.add_argument('-ckpt_epoch', '--ckpt_epoch', type=int, default = None, help='ckpt epoch')
        parser.add_argument('-ckpt_sc', '--ckpt_score', action = 'store_true', help='ckpt name')
        parser.add_argument('-dist', '--dist', action = 'store_true', default = False, help = 'whether to distributed pytorch')
        parser.add_argument('-eval_mode', '--eval_mode', type=str, default = 'quan', help = 'evaluation mode. qual(qualitative)/quan(quantitative)')
        parser.add_argument('-data', '--data', type=str, default = 'DPDD', help = 'dataset to evaluate(DP/pixel)')
        args, _ = parser.parse_known_args()

        config.network = args.network
        config.EVAL.ckpt_name = args.ckpt_name
        config.EVAL.ckpt_abs_name = args.ckpt_abs_name
        config.EVAL.ckpt_epoch = args.ckpt_epoch
        config.EVAL.load_ckpt_by_score = args.ckpt_score

        config.dist = args.dist
        config.EVAL.eval_mode = args.eval_mode
        config.EVAL.data = args.data
        config = set_eval_path(config, config.EVAL.data)

        print(toRed('\tProject : {}'.format(config.project)))
        print(toRed('\tMode : {}'.format(config.mode)))
        print(toRed('\tConfig: {}'.format(config.config)))
        print(toRed('\tNetwork: {}'.format(config.network)))
        print(toRed('\tTrainer: {}'.format(config.trainer)))

        eval(config)
