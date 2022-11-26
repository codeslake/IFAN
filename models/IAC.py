import torch
import torch.nn.functional as Func

def IAC(feat_in, F, N, c, k, is_act_last = True):
    Fs = torch.split(F[:, :N * (c * k * 2), :, :], c * k * 2, dim = 1)
    F_bs = torch.split(F[:, N * (c * k * 2):, :, :], c, dim = 1)
    for i in range(N):
        F1, F2= torch.split(Fs[i], c * k, dim = 1)
        f = SAC(feat_in=feat_in if i == 0 else f, kernel1=F1, kernel2=F2, ksize=k) ## image
        f = f + F_bs[i]

        if i < (N - 1):
            f = Func.leaky_relu(f, 0.1, inplace=True)
        elif is_act_last:
            f = Func.leaky_relu(f, 0.1, inplace=True)

    return f

def SAC(feat_in, kernel1, kernel2, ksize):
    channels = feat_in.size(1)
    N, kernels, H, W = kernel1.size()
    pad = (ksize - 1) // 2

    feat_in = Func.pad(feat_in, (0, 0, pad, pad), mode="replicate")
    feat_in = feat_in.unfold(2, ksize, 1)
    feat_in = feat_in.permute(0, 2, 3, 1, 4).view(N, H, W, channels, -1)

    kernel1 = kernel1.permute(0, 2, 3, 1).view(N, H, W, channels, ksize)
    feat_in = torch.sum(torch.mul(feat_in, kernel1), -1)
    feat_in = feat_in.permute(0, 3, 1, 2)

    feat_in = Func.pad(feat_in, (pad, pad, 0, 0), mode="replicate")
    feat_in = feat_in.unfold(3, ksize, 1)
    feat_in = feat_in.permute(0, 2, 3, 1, 4).view(N, H, W, channels, -1)
    kernel2 = kernel2.permute(0, 2, 3, 1).view(N, H, W, channels, ksize)
    # Should be kernel2 below. The current model checkpoints are trained under this code. We will update the new experiment results soon.
    feat_in = torch.sum(torch.mul(feat_in, kernel1), -1)
    feat_out = feat_in.permute(0, 3, 1, 2)

    return feat_out
