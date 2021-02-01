import torch
import torch.nn as nn
import torch.nn.functional as Func
import collections

class Network(nn.Module):
    def __init__(self, config, ks_in):
        #weight_init: 0.03
        super(Network, self).__init__()
        self.config = config

        ks = config.ks
        self.RBFs = config.RBFs
        self.RBF_num = config.RBF_num
        res_num = config.res_num
        self.wi = config.wi
        self.wiF = config.wiF

        ch4 = config.ch * 4
        self.ch4 = ch4

        self.RBF = nn.Sequential(
            conv(ks_in, ch4, kernel_size=ks),
            resnet_block(ch4, kernel_size=ks, res_num=res_num),
            resnet_block(ch4, kernel_size=ks, res_num=res_num),
            conv(ch4, self.RBF_num * (3 * self.RBFs * 2) + self.RBF_num * 3, kernel_size=3, act = None))

    def set_grad(self, var):
        def hook(grad):
            var.grad = grad

        return hook

    def weights_init(self, m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
            torch.nn.init.xavier_uniform_(m.weight, gain = self.wi)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.InstanceNorm2d:
                if m.weight is not None:
                    torch.nn.init.constant_(m.weight, 1)
                    torch.nn.init.constant_(m.bias, 0)
            elif type(m) == torch.nn.Linear:
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)

    def weights_init_F(self, m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
            torch.nn.init.xavier_uniform_(m.weight, gain = self.wiF)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.InstanceNorm2d:
                if m.weight is not None:
                    torch.nn.init.constant_(m.weight, 1)
                    torch.nn.init.constant_(m.bias, 0)
            elif type(m) == torch.nn.Linear:
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)

    def init_F(self):
        self.RBF.apply(self.weights_init_F)

    def forward(self, S, Filter, is_train = True):
        S = Func.interpolate(S, scale_factor=1/8)
        ## for reblurring
        RBF = self.RBF(Filter)
        RBFs = torch.split(RBF[:, :self.RBF_num * (3 * self.RBFs * 2), :, :], 3 * self.RBFs * 2, dim = 1)
        RBF_bs = torch.split(RBF[:, self.RBF_num * (3 * self.RBFs * 2):, :, :], 3, dim = 1)

        SB = S
        for i in range(self.RBF_num):
            RBF1, RBF2 = torch.split(RBFs[i], 3 * self.RBFs, dim = 1)
            SB = SAC(feat_in=SB, kernel1=RBF1, kernel2=RBF2, ksize=self.RBFs) ## image
            SB = SB + RBF_bs[i]
            if i < (self.RBF_num - 1):
                SB = Func.leaky_relu(SB, 0.1, inplace=True)
        SB = SB + S

        outs = collections.OrderedDict()
        outs['SB'] = SB

        return outs

####################################################################################
def SAC(feat_in, kernel1, kernel2, ksize):
    channels = feat_in.size(1)
    N, kernels, H, W = kernel1.size()
    pad = (ksize - 1) // 2

    feat_in = Func.pad(feat_in, (pad, pad, pad, pad), mode="replicate")
    feat_in = feat_in.unfold(2, ksize, 1).unfold(3, ksize, 1)
    feat_in = feat_in.permute(0, 2, 3, 1, 5, 4).reshape(N, H, W, channels, -1)

    kernel1 = kernel1.permute(0, 2, 3, 1).reshape(N, H, W, channels, ksize, 1)
    kernel2 = kernel2.permute(0, 2, 3, 1).reshape(N, H, W, channels, 1, ksize)
    kernel = torch.matmul(kernel1.contiguous(), kernel2.contiguous())
    kernel = kernel.permute(0, 1, 2, 3, 5, 4).reshape(N, H, W, channels, -1)

    feat_out = torch.sum(torch.mul(feat_in.contiguous(), kernel.contiguous()), -1)
    feat_out = feat_out.permute(0, 3, 1, 2).contiguous()

    return feat_out

def conv(in_channels, out_channels, kernel_size=3, stride=1,dilation=1, bias=True, act = 'LeakyReLU'):
    if act is not None:
        if act == 'LeakyReLU':
            act_ = nn.LeakyReLU(0.1,inplace=True)
        elif act == 'Sigmoid':
            act_ = nn.Sigmoid()
        elif act == 'Tanh':
            act_ = nn.Tanh()

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias),
            act_
        )
    else:
        return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias)

def resnet_block(in_channels,  kernel_size=3, dilation=[1,1], bias=True, res_num=1):
    return ResnetBlock(in_channels, kernel_size, dilation, bias=bias, res_num=res_num)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, bias, res_num):
        super(ResnetBlock, self).__init__()
        self.res_num = res_num
        self.stem = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[0], padding=((kernel_size-1)//2)*dilation[0], bias=bias),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[1], padding=((kernel_size-1)//2)*dilation[1], bias=bias),
            ) for i in range(res_num)
        ])
    def forward(self, x):

        if self.res_num > 1:
            temp = x

        for i in range(self.res_num):
            xx = self.stem[i](x)
            x = x + xx
            x = Func.leaky_relu(x, 0.1, inplace=True)

        if self.res_num > 1:
            x = x + temp

        return x

