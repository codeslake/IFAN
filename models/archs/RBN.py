import torch
import torch.nn as nn
import torch.nn.functional as Func
import collections

from models.nn_common import conv, upconv, resnet_block
from models.IAC import IAC

class Network(nn.Module):
    def __init__(self, config, ks_in):
        #weight_init: 0.03
        super(Network, self).__init__()
        self.config = config

        ks = config.ks
        self.RBFs = config.RBFs
        self.RBF_num = config.RBF_num
        res_num = config.res_num
        self.wiRF = config.wiRF

        ch4 = config.ch * 4
        self.ch4 = ch4

        self.RBF = nn.Sequential(
            conv(ks_in, ch4, kernel_size=ks),
            resnet_block(ch4, kernel_size=ks, res_num=res_num),
            resnet_block(ch4, kernel_size=ks, res_num=res_num),
            conv(ch4, self.RBF_num * (3 * self.RBFs * 2) + self.RBF_num * 3, kernel_size=3, act = None))

    def weights_init_F(self, m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
            torch.nn.init.xavier_uniform_(m.weight, gain = self.wiRF)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def init_F(self):
        self.RBF.apply(self.weights_init_F)

##########################################################################
    def forward(self, S, Filter, is_train = True):
        S = Func.interpolate(S, scale_factor=1/8)
        SB = S

        ## for reblurring
        RBF = self.RBF(Filter)

        SB = IAC(SB, RBF, self.RBF_num, 3, self.RBFs, is_act_last = False)

        SB = SB + S

        outs = collections.OrderedDict()
        outs['SB'] = SB

        return outs
