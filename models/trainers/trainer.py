import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import numpy as np
import collections
import torch_optimizer as optim
import torch.nn.utils as torch_utils

from utils import *
from data_loader.utils import *
from models.utils import *
from models.baseModel import baseModel
import models.archs.LPIPS as LPIPS
from models.archs.RBN import Network as reblurNet

from data_loader.DP_datasets_lr_aug import datasets

import models.trainers.lr_scheduler as lr_scheduler

from data_loader.data_sampler import DistIterSampler

from ptflops import get_model_complexity_info
import importlib
from shutil import copy2
import os

class Model(baseModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.rank = torch.distributed.get_rank() if config.dist else -1
        self.ws = torch.distributed.get_world_size() if config.dist else 1
        self.device = config.device

        ### NETWORKS ###
        ## main network
        lib = importlib.import_module('models.archs.{}'.format(config.network))
        if self.is_train and self.config.resume is None or self.is_train and os.path.exists('./models/archs/{}.py'.format(config.network)):
            copy2('./models/archs/{}.py'.format(config.network), self.config.LOG_DIR.offset)

        if self.rank <= 0 : print(toGreen('Loading Model...'))
        self.network = DeblurNet(config, lib).to(self.device)

        ## LPIPS network
        if self.is_train:
            if self.config.is_amp:
                self.scaler = torch.cuda.amp.GradScaler()

            if self.rank <= 0: print(toRed('\tinitializing LPIPS'))
            if config.cuda:
                if config.dist:
                    ## download pretrined checkpoint from 0th process
                    if self.rank <= 0:
                        self.LPIPS = LPIPS.PerceptualLoss(model='net-lin',net='alex', gpu_ids = [torch.cuda.current_device()]).to(torch.device('cuda'))
                    dist.barrier()
                    if self.rank > 0:
                        self.LPIPS = LPIPS.PerceptualLoss(model='net-lin',net='alex', gpu_ids = [torch.cuda.current_device()]).to(torch.device('cuda'))

                else:
                    self.LPIPS = LPIPS.PerceptualLoss(model='net-lin',net='alex', gpu_ids = [torch.cuda.current_device()]).to(torch.device('cuda'))
            else:
                self.LPIPS = LPIPS.PerceptualLoss(model='net-lin',net='alex', use_gpu=False).to(self.device)

            for param in self.LPIPS.parameters():
                param.requires_grad_(False)

        ### INIT for training ###
        if self.is_train:
            self.itr_global = {'train': 0, 'valid': 0}
            self.itr_inc = {'train': 1, 'valid': 1} # iteration increase factor (for video processing, it might be higher than 1)
            self.network.init()
            self._set_optim()
            self._set_lr_scheduler()
            self._set_dataloader()

            if config.is_verbose:
                for name, param in self.network.named_parameters():
                    if self.rank <= 0: print(name, ', ', param.requires_grad)

        ### PROFILE ###
        if self.rank <= 0:
            with torch.no_grad():
                Macs,params = get_model_complexity_info(self.network.Network, (1, 3, 720, 1280), input_constructor = self.network.input_constructor, as_strings=False,print_per_layer_stat=config.is_verbose)

        ### DDP ###
        if config.cuda:
            if config.dist:
                if self.rank <= 0: print(toGreen('Building Dist Parallel Model...'))
                self.network = DDP(self.network, device_ids=[torch.cuda.current_device()], output_device=torch.cuda.current_device(), broadcast_buffers=True, find_unused_parameters=False)
            else:
                self.network = DP(self.network).to(torch.device('cuda'))

            if self.rank <= 0:
                print(toGreen('Computing model complexity...'))
                print(toRed('\t{:<30}  {:<8} B'.format('Computational complexity (Macs): ', Macs / 1000 ** 3 )))
                print(toRed('\t{:<30}  {:<8} M'.format('Number of parameters: ',params / 1000 ** 2, '\n')))
                if self.is_train:
                    with open(config.LOG_DIR.offset + '/cost.txt', 'w') as f:
                        f.write('{:<30}  {:<8} B\n'.format('Computational complexity (Macs): ', Macs / 1000 ** 3 ))
                        f.write('{:<30}  {:<8} M'.format('Number of parameters: ',params / 1000 ** 2))
                        f.close()

    def get_itr_per_epoch(self, state):
        if state == 'train':
            return len(self.data_loader_train) * self.itr_inc[state]
        else:
            return len(self.data_loader_eval) * self.itr_inc[state]

    def _set_loss(self, lr = None):
        if self.rank <= 0: print(toGreen('Building Loss...'))
        self.MSE = torch.nn.MSELoss().to(self.device)
        self.MAE = torch.nn.L1Loss().to(self.device)
        self.CSE = torch.nn.CrossEntropyLoss(reduction='none').to(self.device)
        self.MSE_sum = torch.nn.MSELoss(reduction = 'sum').to(self.device)

    def _set_optim(self, lr = None):
        if self.rank <= 0: print(toGreen('Building Optim...'))
        self._set_loss()
        lr = self.config.lr_init if lr is None else lr

        self.optimizer = optim.RAdam([
            {'params': self.network.parameters(), 'lr': self.config.lr_init, 'initial_lr': self.config.lr_init}
            ], eps= 1e-8, weight_decay=0.01, lr=lr, betas=(self.config.beta1, 0.999))

        self.optimizers.append(self.optimizer)

    def _set_lr_scheduler(self):
        if self.config.LRS == 'CA':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingLR_Restart(
                        optimizer, self.config.T_period, eta_min= self.config.eta_min,
                        restarts= self.config.restarts, weights= self.config.restart_weights))
        elif self.config.LRS == 'LD':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.LR_decay(
                        optimizer, decay_period = self.config.decay_period,
                        decay_rate = self.config.decay_rate))

    def _set_dataloader(self):
        if self.rank <= 0: print(toGreen('Loading Data Loader...'))

        self.sampler_train = None
        self.sampler_eval = None

        self.dataset_train = datasets(self.config, is_train = True)
        self.dataset_eval = datasets(self.config, is_train = False)

        if self.config.dist == True:
            self.sampler_train = DistIterSampler(self.dataset_train, self.ws, self.rank)
            self.sampler_eval = DistIterSampler(self.dataset_eval, self.ws, self.rank, is_train=False)
        else:
            self.sampler_train = None
            self.sampler_eval = None

        if self.is_train:
            self.data_loader_train = self._create_dataloader(self.dataset_train, sampler = self.sampler_train, is_train = True)
            self.data_loader_eval = self._create_dataloader(self.dataset_eval, sampler = self.sampler_eval, is_train = False)

    def _update(self, errs, warmup_itr = -1):
        lr = None
        self.optimizer.zero_grad()
        errs['total'].backward()

        torch_utils.clip_grad_norm_(self.network.parameters(), self.config.gc)
        self.optimizer.step()
        lr = self._update_learning_rate(self.itr_global['train'], warmup_itr)

        return lr

    def _update_amp(self, errs, warmup_itr = -1):
        lr = None
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(errs['total']).backward()

        self.scaler.unscale_(self.optimizer)
        torch_utils.clip_grad_norm_(self.network.parameters(), self.config.gc)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        lr = self._update_learning_rate(self.itr_global['train'], warmup_itr)

        return lr

    ########################### Edit from here for training/testing scheme ###############################
    def _set_results(self, inputs, outs, errs, lr, norm=1):

        # save visuals (inputs)
        if self.config.save_sample:
            self.results['vis'] = collections.OrderedDict()
            if 'dual' not in self.config.mode:
                self.results['vis']['C'] = inputs['c']
            else:
                self.results['vis']['L'] = inputs['l']
                self.results['vis']['R'] = inputs['r']

            self.results['vis']['result'] = outs['result']
            self.results['vis']['GT'] = inputs['gt']

            if 'D' in self.config.mode or 'IFAN' in self.config.mode:
                self.results['vis']['f_R'] = F.interpolate(inputs['r'], scale_factor=1/8, mode='area')
                self.results['vis']['f_R_w'] = outs['f_R_w']
                self.results['vis']['f_R_gt'] = F.interpolate(inputs['l'], scale_factor=1/8, mode='area')
            if 'R' in self.config.mode or 'IFAN' in self.config.mode:
                self.results['vis']['S'] = F.interpolate(inputs['gt'], scale_factor=1/8, mode='area')
                self.results['vis']['SB'] = outs['SB']
                self.results['vis']['B'] = F.interpolate(inputs['c'], scale_factor=1/8, mode='area')

        ## essential ##
        self.results['errs'] = errs
        self.results['norm'] = norm
        self.results['lr'] = lr

    def _get_results(self, C, R, L, GT, is_train):
        if 'dual' not in  self.config.mode:
            L = L if is_train or self.config.save_sample else None
            R = R if is_train or self.config.save_sample else None

        if self.config.is_amp:
            with torch.cuda.amp.autocast():
                outs = self.network(C, R, L, GT, is_train=is_train)
        else:
            outs = self.network(C, R, L, GT, is_train=is_train)

        ## loss
        if self.config.is_train:
            errs = collections.OrderedDict()
            if 'result' in outs.keys():
                errs['total'] = 0.
                # deblur loss
                errs['image'] = self.MSE(outs['result'], GT)
                errs['total'] = errs['total'] + errs['image']

            if is_train:
                if 'LPIPS' in self.config.mode:
                    # explodes when you use DDP
                    dist = self.LPIPS.forward(outs['result'] * 2. - 1., GT * 2. - 1.) #imge range [-1, 1]
                    with torch.no_grad():
                        errs['LPIPS'] = torch.mean(dist) # flow log
                    errs['LPIPS_MSE'] = 1e-1 * self.MSE(torch.zeros_like(dist).to(self.device), dist)
                    errs['total'] = errs['total'] + errs['LPIPS_MSE']

                if ('D' in self.config.mode or 'IFAN' in self.config.mode) and 'f_R_w' in outs.keys():
                    errs['dme'] = self.MSE(outs['f_R_w'], outs['f_L'])
                    errs['total'] = errs['total'] + errs['dme']

                if 'R' in self.config.mode or 'IFAN' in self.config.mode:
                    C_down = F.interpolate(C, scale_factor=1/8, mode='area')
                    errs['reblur'] = self.MSE(outs['SB'], C_down)
                    errs['total'] = errs['total'] + errs['reblur']

            else:
                errs['psnr'] = get_psnr2(outs['result'], GT)
                dist = self.LPIPS.forward(outs['result'] * 2. - 1., GT * 2. - 1.) #imge range [-1, 1]
                errs['LPIPS'] = torch.mean(dist) # flow log

            return errs, outs
        else:
            return outs

    def iteration(self, inputs, epoch, max_epoch, is_train):
        # init for logging
        state = 'train' if is_train else 'valid'
        self.itr_global[state] += self.itr_inc[state]

        # dataloader / loading data / mod crop
        inputs['c'] = refine_image_pt(inputs['c'].to(self.device), self.config.refine_val)
        inputs['r'] = refine_image_pt(inputs['r'].to(self.device), self.config.refine_val)
        inputs['l'] = refine_image_pt(inputs['l'].to(self.device), self.config.refine_val)
        inputs['gt'] = refine_image_pt(inputs['gt'].to(self.device), self.config.refine_val)

        C = inputs['c']
        R = inputs['r']
        L = inputs['l']
        GT = inputs['gt']

        # run network / get results and losses / update network, get learning rate
        errs, outs = self._get_results(C, R, L, GT, is_train)
        if self.config.is_amp:
            lr = self._update_amp(errs, self.config.warmup_itr) if is_train else None
        else:
            lr = self._update(errs, self.config.warmup_itr) if is_train else None


        # set results for the log
        outs_ = collections.OrderedDict()
        for k, v in outs.items():
            outs_[k] = v.clone().detach()
        errs_ = collections.OrderedDict()
        for k, v in errs.items():
            errs_[k] = v.clone().detach()
        norm, _, _, _ = outs_['result'].shape
        self._set_results(inputs, outs_, errs_, lr, norm)

class DeblurNet(nn.Module):
    def __init__(self, config, lib):
        super(DeblurNet, self).__init__()
        self.rank = torch.distributed.get_rank() if config.dist else -1

        self.config = config
        self.device = config.device
        self.Network = lib.Network(config)
        if self.rank <= 0: print(toRed('\tinitializing deblurring network'))

        if self.config.is_train and ('R' in config.mode or 'IFAN' in config.mode):
            if self.rank <= 0: print(toRed('\tinitializing RBN'))
            self.reblurNet = reblurNet(config, self.Network.kernel_dim)

    def weights_init(self, m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
            torch.nn.init.xavier_uniform_(m.weight, gain = self.config.wi)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.InstanceNorm2d:
            if m.weight is not None:
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
        elif type(m) == torch.nn.Linear:
            torch.nn.init.normal_(m.weight, 0, 0.01)
            torch.nn.init.constant_(m.bias, 0)

    def init(self):
        self.Network.apply(self.weights_init)
        self.Network.init_F()
        if self.config.is_train and ('R' in self.config.mode or 'IFAN' in self.config.mode):
            self.reblurNet.apply(self.weights_init)
            self.reblurNet.init_F()

    def input_constructor(self, res):
        b, c, h, w = res[:]

        C = torch.FloatTensor(np.random.randn(b, c, h, w)).to(self.device)

        return {'C':C, 'R':C, 'L':C}

    #####################################################
    def forward(self, C, R=None, L=None, GT=None, is_train=False):
        is_train = is_train or self.config.save_sample and self.config.is_train

        outs = self.Network(C, R, L, is_train)

        if is_train and ('R' in self.config.mode or 'IFAN' in self.config.mode):
            outs_reblur = self.reblurNet(GT, outs['Filter'])
            outs.update(outs_reblur)

        return outs

