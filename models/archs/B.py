import torch
import torch.nn as nn
import collections
from models.nn_common import conv, upconv, resnet_block

class Network(nn.Module):
    def __init__(self, config):
        #weight_init: 0.03
        super(Network, self).__init__()
        try:
            self.rank = torch.distributed.get_rank()
        except Exception as ex:
            self.rank = 0

        ks = config.ks
        self.Fs = config.Fs
        res_num = config.res_num

        ch1 = config.ch 
        ch2 = ch1 * 2 
        ch3 = ch1 * 4
        ch4 = ch1 * 4
        self.ch4 = ch4

        ###################################
        # Feature Extractor - Reconstructor
        ###################################
        # feature extractor
        self.conv1_1 = conv(3, ch1, kernel_size=ks, stride=1)
        self.conv1_2 = conv(ch1, ch1, kernel_size=ks, stride=1)
        self.conv1_3 = conv(ch1, ch1, kernel_size=ks, stride=1)

        self.conv2_1 = conv(ch1, ch2, kernel_size=ks, stride=2)
        self.conv2_2 = conv(ch2, ch2, kernel_size=ks, stride=1)
        self.conv2_3 = conv(ch2, ch2, kernel_size=ks, stride=1)

        self.conv3_1 = conv(ch2, ch3, kernel_size=ks, stride=2)
        self.conv3_2 = conv(ch3, ch3, kernel_size=ks, stride=1)
        self.conv3_3 = conv(ch3, ch3, kernel_size=ks, stride=1)

        self.conv4_1 = conv(ch3, ch4, kernel_size=ks, stride=2)
        self.conv4_2 = conv(ch4, ch4, kernel_size=ks, stride=1)
        self.conv4_3 = conv(ch4, ch4, kernel_size=ks, stride=1)

        self.conv4_4 = nn.Sequential(
            conv(2 * ch4, ch4, kernel_size=ks),
            resnet_block(ch4, kernel_size=ks, res_num=res_num),
            resnet_block(ch4, kernel_size=ks, res_num=res_num),
            conv(ch4, ch4, kernel_size=ks))

        # reconstructor
        self.conv_res = nn.Sequential(
            conv(ch4, ch4, kernel_size=ks),
            resnet_block(ch4, kernel_size=ks, res_num=19),
            conv(ch4, ch4, kernel_size=ks))

        self.upconv3_u = upconv(ch4, ch3)
        self.upconv3_1 = resnet_block(ch3, kernel_size=ks, res_num=1)
        self.upconv3_2 = resnet_block(ch3, kernel_size=ks, res_num=1)

        self.upconv2_u = upconv(ch3, ch2)
        self.upconv2_1 = resnet_block(ch2, kernel_size=ks, res_num=1)
        self.upconv2_2 = resnet_block(ch2, kernel_size=ks, res_num=1)

        self.upconv1_u = upconv(ch2, ch1)
        self.upconv1_1 = resnet_block(ch1, kernel_size=ks, res_num=1)
        self.upconv1_2 = resnet_block(ch1, kernel_size=ks, res_num=1)

        self.out_res = conv(ch1, 3, kernel_size=ks)
        ###################################

        ###################################
        # IFAN
        ###################################
        # filter encoder

        self.kconv1_1 = conv(3, ch1, kernel_size=ks, stride=1)
        self.kconv1_2 = conv(ch1, ch1, kernel_size=ks, stride=1)
        self.kconv1_3 = conv(ch1, ch1, kernel_size=ks, stride=1)

        self.kconv2_1 = conv(ch1, ch2, kernel_size=ks, stride=2)
        self.kconv2_2 = conv(ch2, ch2, kernel_size=ks, stride=1)
        self.kconv2_3 = conv(ch2, ch2, kernel_size=ks, stride=1)

        self.kconv3_1 = conv(ch2, ch3, kernel_size=ks, stride=2)
        self.kconv3_2 = conv(ch3, ch3, kernel_size=ks, stride=1)
        self.kconv3_3 = conv(ch3, ch3, kernel_size=ks, stride=1)

        self.kconv4_1 = conv(ch3, ch4, kernel_size=ks, stride=2)
        self.kconv4_2 = conv(ch4, ch4, kernel_size=ks, stride=1)
        self.kconv4_3 = conv(ch4, ch4, kernel_size=ks, stride=1)

    def init_F(self):
        return 

##########################################################################
    def forward(self, C, R=None, L=None, is_train=False):
        # feature extractor
        f1 = self.conv1_3(self.conv1_2(self.conv1_1(C)))
        f2 = self.conv2_3(self.conv2_2(self.conv2_1(f1)))
        f3 = self.conv3_3(self.conv3_2(self.conv3_1(f2)))
        f_C = self.conv4_3(self.conv4_2(self.conv4_1(f3)))

        #filter encoder
        f = self.kconv1_3(self.kconv1_2(self.kconv1_1(C)))
        f = self.kconv2_3(self.kconv2_2(self.kconv2_1(f)))
        f = self.kconv3_3(self.kconv3_2(self.kconv3_1(f)))
        f = self.kconv4_3(self.kconv4_2(self.kconv4_1(f)))

        f = self.conv4_4(torch.cat([f, f_C], 1))
        f = self.conv_res(f)

        # reconstructor
        f = self.upconv3_u(f) + f3
        f = self.upconv3_2(self.upconv3_1(f))

        f = self.upconv2_u(f) + f2
        f = self.upconv2_2(self.upconv2_1(f))

        f = self.upconv1_u(f) + f1
        f = self.upconv1_2(self.upconv1_1(f))

        out = self.out_res(f) + C

        # results
        outs = collections.OrderedDict()
        outs['result'] = out

        return outs
