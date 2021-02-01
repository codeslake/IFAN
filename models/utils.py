import torch
import torch.nn.functional as F
import numpy as np
import math
import time

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if classname.find('KernelConv2D') == -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.04)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.04)
        torch.nn.init.constant_(m.bias.data, 0)

def adjust_learning_rate(optimizer, epoch, decay_rate, decay_every, lr_init):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lrs = []
    for param_group in optimizer.param_groups:

        lr = param_group['lr_init'] * (decay_rate ** (epoch // decay_every))
        param_group['lr'] = lr
        lrs.append(lr)

    return lrs

def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

def feat_vis(tensor):
    tensor = tensor.clone().permute(1, 0, 2, 3)
    for b in range(tensor.size()[0]):
        tensor[b] -= tensor[b].min()
        tensor[b] /= tensor[b].max()

    return tensor

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def get_psnr(img1, img2):
    return 20.0*math.log10(255.0 / math.sqrt(np.mean( ((img1.cpu().numpy() * 255.).round() - (img2.cpu().numpy() * 255.).round()) ** 2 )))
    # img1 = tensor2img(img1)
    # img2 = tensor2img(img2)

    # return calculate_psnr(img1, img2)

#from DPDD code
def get_psnr2(img1, img2, PIXEL_MAX=1.0):
    mse_ = torch.mean( (img1 - img2) ** 2 )
    return 10 * torch.log10(PIXEL_MAX / mse_)

    # return calculate_psnr(img1, img2)

Backward_tensorGrid = {}
def warp(tensorInput, tensorFlow, padding_mode = 'zeros'):
    if str(tensorFlow.size()) not in Backward_tensorGrid:
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

        Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([ tensorHorizontal, tensorVertical ], 1).to(torch.device('cuda'))

    tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)
    # tensorFlow = torch.cat([ 2.0 * (tensorFlow[:, 0:1, :, :] / (tensorInput.size(3) - 1.0)) - 1.0 , 2.0 * (tensorFlow[:, 1:2, :, :] / (tensorInput.size(2) - 1.0)) - 1.0  ], 1)

    return torch.nn.functional.grid_sample(input=tensorInput, grid=(Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode=padding_mode, align_corners = True)

def DOF(tensorInput, DoF, padding_mode = 'zeros'):
    tensorFlow = DoF[:, :2, :, :]
    alpha1 = DoF[:, 2:5, :, :]
    alpha2 = DoF[:, 5:, :, :]

    if str(tensorFlow.size()) not in Backward_tensorGrid:
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

        Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([ tensorHorizontal, tensorVertical ], 1).to(torch.device('cuda'))

    tensorFlow1 = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), torch.zeros_like(tensorFlow[:, 0:1, :, :]).to(torch.device('cuda')) ], 1)
    sample1 = torch.nn.functional.grid_sample(input=tensorInput, grid=(Backward_tensorGrid[str(tensorFlow1.size())] + tensorFlow1).permute(0, 2, 3, 1), mode='bilinear', padding_mode=padding_mode, align_corners = True)

    tensorFlow2 = torch.cat([ tensorFlow[:, 1:2, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), torch.zeros_like(tensorFlow[:, 1:2, :, :]).to(torch.device('cuda')) ], 1)
    sample2 = torch.nn.functional.grid_sample(input=tensorInput, grid=(Backward_tensorGrid[str(tensorFlow2.size())] + tensorFlow2).permute(0, 2, 3, 1), mode='bilinear', padding_mode=padding_mode, align_corners = True)

    return (sample1 * torch.sigmoid(alpha1) + sample2 * torch.sigmoid(alpha2)) / 2.

DPD_zero = {}
def DPD(tensorInput, tensorFlow, padding_mode = 'zeros'):
    if str(tensorFlow.size()) not in Backward_tensorGrid:
        DPD_zero[str(tensorFlow.size())] = torch.zeros_like(tensorFlow[:, 0:1, :, :]).to(torch.device('cuda'))
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

        Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([ tensorHorizontal, tensorVertical ], 1).to(torch.device('cuda'))

    DPDM = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),  DPD_zero[str(tensorFlow.size())]], 1)
    return torch.nn.functional.grid_sample(input=tensorInput, grid=(Backward_tensorGrid[str(tensorFlow.size())] + DPDM).permute(0, 2, 3, 1), mode='bilinear', padding_mode=padding_mode, align_corners = True)

def upsample(inp, h = None, w = None, mode = 'bilinear'):
    # if h is None or w is None:
    return F.interpolate(input=inp, size=(int(h), int(w)), mode=mode)
    # elif scale_factor is not None:
    #     return F.interpolate(input=inp, scale_factor=scale_factor, mode='bilinear', align_corners=False)
            
def feature_matching(t_image, t_image2, corr_, pool, flows = None, scales = None):
    if flows is not None:
        size = t_image2.size()
        for idx, flow in enumerate(flows):
            flow = upsample(flow, h = size[2], w = size[3], mode = 'nearest') * scales[idx]
            assert(t_image2.size()[2]==flow.size()[2] and t_image2.size()[3] == flow.size(3))
            t_image2 = warp(t_image2, flow, 'zeros') #zeros, reflection

    shape = t_image.size()
    channel = shape[1]
    l2norm = torch.sqrt(torch.sum(torch.mul(t_image,t_image),1,keepdim=True))
    l2norm2 = torch.sqrt(torch.sum(torch.mul(t_image2,t_image2),1,keepdim=True))
    corr = channel * corr_(t_image / (l2norm + 1e-8), t_image2 / (l2norm2 + 1e-8))
    corr = pool(corr)
    # corr = torch.nn.AvgPool2d(corr,[5,5],"AVG",padding='SAME',strides=[1,1],data_format='NHWC')
    matching_index = torch.argmax(corr, dim=1).type(torch.cuda.FloatTensor)

    # rank = torch.distributed.get_rank()
    # if rank<=0: print('(MD:', corr_.max_displacement, ')', 'MI: ', matching_index.min(), matching_index.max())
    # if rank<=0: print('corr: ', corr.max(), corr.min(), corr.std())
    # if rank<=0: print('F1: ', (t_image / (l2norm + 1e-8)).sum(), ' F2: ', (t_image / (l2norm2 + 1e-8)).sum())
    # print('(MD:', corr_.max_displacement, ')', 'MI: ', matching_index.min(), matching_index.max())
    # print('corr: ', corr.max(), corr.min(), corr.std())

    # stride2: inner stride
    kernel_size = corr_.max_displacement * 2 + 1
    half_ks = np.floor(kernel_size/(corr_.stride2*2))
    y = ((matching_index//np.floor(kernel_size/float(corr_.stride2)+0.5)) - half_ks) * corr_.stride2
    x = ((matching_index%np.floor(kernel_size/float(corr_.stride2)+0.5)) - half_ks) * corr_.stride2

    flow = torch.cat((torch.unsqueeze(x,1),torch.unsqueeze(y,1)), axis=1)
    n = warp(t_image2, flow)
    return n, corr, flow

def FM(F1, F2, F3, corr_, pool, scale = None):
    shape = F1.size()
    channel = shape[1]
    l2norm = torch.sqrt(torch.sum(torch.mul(F1,F1),1,keepdim=True))
    l2norm2 = torch.sqrt(torch.sum(torch.mul(F2,F2),1,keepdim=True))
    # prev_time = time.time()#
    corr = channel * corr_(F1 / (l2norm + 1e-8), F2 / (l2norm2 + 1e-8))
    # print('1: ', time.time() - prev_time)#

    # prev_time = time.time()#
    corr = pool(corr)
    matching_index = torch.argmax(corr, dim=1).type(torch.cuda.FloatTensor)
    # print('2: ', time.time() - prev_time)

    # rank = torch.distributed.get_rank()
    # if rank<=0: print('(MD:', corr_.max_displacement, ')', 'MI: ', matching_index.min(), matching_index.max())
    # if rank<=0: print('corr: ', corr.max(), corr.min(), corr.std())
    # if rank<=0: print('F1: ', (F / (l2norm + 1e-8)).sum(), ' F2: ', (F / (l2norm2 + 1e-8)).sum())
    # print('(MD:', corr_.max_displacement, ')', 'MI: ', matching_index.min(), matching_index.max())
    # print('corr: ', corr.max(), corr.min(), corr.std())

    # stride2: inner stride
    # prev_time = time.time()#
    kernel_size = corr_.max_displacement * 2 + 1
    half_ks = np.floor(kernel_size/(corr_.stride2*2))
    y = ((matching_index//np.floor(kernel_size/float(corr_.stride2)+0.5)) - half_ks) * corr_.stride2
    x = ((matching_index%np.floor(kernel_size/float(corr_.stride2)+0.5)) - half_ks) * corr_.stride2
    # print('3: ', time.time() - prev_time)#

    flow = torch.cat((torch.unsqueeze(x,1),torch.unsqueeze(y,1)), axis=1)
    if scale is not None:
        shape = F3.size()
        flow = upsample(flow, shape[2], shape[3], 'nearest') * scale

    n = warp(F3, flow)
    return n, corr, flow

def FM_no_warp(F1, F2, corr_, pool, scale = None):
    shape = F1.size()
    channel = shape[1]
    l2norm = torch.sqrt(torch.sum(torch.mul(F1,F1),1,keepdim=True))
    l2norm2 = torch.sqrt(torch.sum(torch.mul(F2,F2),1,keepdim=True))
    corr = channel * corr_(F1 / (l2norm + 1e-8), F2 / (l2norm2 + 1e-8))
    corr = pool(corr)
    matching_index = torch.argmax(corr, dim=1).type(torch.cuda.FloatTensor)

    # rank = torch.distributed.get_rank()
    # if rank<=0: print('(MD:', corr_.max_displacement, ')', 'MI: ', matching_index.min(), matching_index.max())
    # if rank<=0: print('corr: ', corr.max(), corr.min(), corr.std())
    # if rank<=0: print('F1: ', (F / (l2norm + 1e-8)).sum(), ' F2: ', (F / (l2norm2 + 1e-8)).sum())
    # print('(MD:', corr_.max_displacement, ')', 'MI: ', matching_index.min(), matching_index.max())
    # print('corr: ', corr.max(), corr.min(), corr.std())

    # stride2: inner stride
    kernel_size = corr_.max_displacement * 2 + 1
    half_ks = np.floor(kernel_size/(corr_.stride2*2))
    y = ((matching_index//np.floor(kernel_size/float(corr_.stride2)+0.5)) - half_ks) * corr_.stride2
    x = ((matching_index%np.floor(kernel_size/float(corr_.stride2)+0.5)) - half_ks) * corr_.stride2

    flow = torch.cat((torch.unsqueeze(x,1),torch.unsqueeze(y,1)), axis=1)
    if scale is not None:
        shape = F2.size()
        flow = upsample(flow, shape[2] * scale, shape[3] * scale, 'nearest') * scale

    return corr, flow


def FM_SA(F1, F2, F3, corr_, pool, resample, scale = None):
    shape = F1.size()
    channel = shape[1]
    l2norm = torch.sqrt(torch.sum(torch.mul(F1,F1),1,keepdim=True))
    l2norm2 = torch.sqrt(torch.sum(torch.mul(F2,F2),1,keepdim=True))
    corr = channel * corr_(F1 / (l2norm + 1e-8), F2 / (l2norm2 + 1e-8))
    corr = pool(corr)
    sorted, ind = torch.sort(corr[0, :, 32, 32], descending=True)
    # print(sorted[:10])
    # print(corr[0, :, 32, 32].reshape(21, 21))
    # print(corr[0, :, 32, 32].sum())
    # print(ind.reshape(21, 21))

    beta = 500
    B, C, H, W = corr.size()
    chw = int(math.sqrt(C))

    corr_ = corr.permute(0, 2, 3, 1)
    corr_ = corr_.reshape((B*H*W, C))
    corr_ = F.softmax(beta*corr_, dim = 1)
    corr_ = corr_.reshape((B*H*W, 1, chw, chw))
    corr_ = pool(corr_)
    corr_ = corr_.reshape((B*H*W, chw, chw))

    ind = torch.arange(-int(math.floor(chw/2)), int(math.ceil(chw/2))).to(torch.device('cuda'))

    x = torch.sum(torch.sum(corr_, 1) * ind, 1)
    x = x.view((B, H, W))
    y = torch.sum(torch.sum(corr_, 2) * ind, 1)
    y = y.view((B, H, W))

    flow = torch.cat((torch.unsqueeze(x,1),torch.unsqueeze(y,1)), axis=1)
    if scale is not None:
        shape = F3.size()
        flow = upsample(flow, shape[2], shape[3], 'area') * scale

    n = warp(F3, flow)

    return n, corr, flow

def soft_argmax(voxels):
    """
    Arguments: voxel patch in shape (batch_size, channel, H, W, depth)
    Return: 3D coordinates in shape (batch_size, channel, 3)
    """
    # alpha is here to make the largest element really big, so it
    # would become very close to 1 after softmax
    alpha = 1000.0 
    N,C,H,W,D = voxels.shape
    soft_max = nn.functional.softmax(voxels.view(N,C,-1)*alpha,dim=2)
    soft_max = soft_max.view(voxels.shape)
    indices_kernel = torch.arange(start=0,end=H*W*D).unsqueeze(0)
    indices_kernel = indices_kernel.view((H,W,D))
    conv = soft_max*indices_kernel
    indices = conv.sum(2).sum(2).sum(2)
    z = indices%D
    y = (indices/D).floor()%W
    x = (((indices/D).floor())/W).floor()%H
    coords = torch.stack([x,y,z],dim=2)
    return coords

def flow2img(flow_data):
    """
    convert optical flow into color image
    :param flow_data:
    :return: color image
    """
    # print(flow_data.shape)
    # print(type(flow_data))
    u = flow_data[:, :, 1]
    v = flow_data[:, :, 0]

    UNKNOW_FLOW_THRESHOLD = 1e7
    pr1 = abs(u) > UNKNOW_FLOW_THRESHOLD
    pr2 = abs(v) > UNKNOW_FLOW_THRESHOLD
    idx_unknown = (pr1 | pr2)
    u[idx_unknown] = v[idx_unknown] = 0

    # get max value in each direction
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxu = max(maxu, np.max(u))
    maxv = max(maxv, np.max(v))
    minu = min(minu, np.min(u))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))
    u = u / maxrad + np.finfo(float).eps
    v = v / maxrad + np.finfo(float).eps

    img = compute_color(u, v)

    idx = np.repeat(idx_unknown[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: horizontal optical flow
    :param v: vertical optical flow
    :return:
    """

    height, width = u.shape
    img = np.zeros((height, width, 3))

    NAN_idx = np.isnan(u) | np.isnan(v)
    u[NAN_idx] = v[NAN_idx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - NAN_idx)))

    return img

def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel
