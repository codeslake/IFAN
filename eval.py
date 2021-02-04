import torch
import torchvision.utils as vutils
import torch.nn.functional as F

import os
import sys
import datetime
import gc
from pathlib import Path

import numpy as np
import cv2
import math
from sklearn.metrics import mean_absolute_error
from skimage.metrics import structural_similarity
import collections

from utils import *
from data_loader.utils import refine_image, read_frame
from ckpt_manager import CKPT_Manager

from models import create_model
import models.archs.LPIPS as LPIPS

def mae(img1, img2):
    mae_0=mean_absolute_error(img1[:,:,0], img2[:,:,0],
                              multioutput='uniform_average')
    mae_1=mean_absolute_error(img1[:,:,1], img2[:,:,1],
                              multioutput='uniform_average')
    mae_2=mean_absolute_error(img1[:,:,2], img2[:,:,2],
                              multioutput='uniform_average')
    return np.mean([mae_0,mae_1,mae_2])

def ssim(img1, img2, PIXEL_MAX = 1.0):
    return structural_similarity(img1, img2, data_range=PIXEL_MAX, multichannel=True)

def psnr(img1, img2, PIXEL_MAX = 1.0):
    mse_ = np.mean( (img1 - img2) ** 2 )
    return 10 * math.log10(PIXEL_MAX / mse_)

def init(config, mode = 'deblur'):
    date = datetime.datetime.now().strftime('%Y_%m_%d_%H%M')

    model = create_model(config)
    network = model.get_network().eval()

    ckpt_manager = CKPT_Manager(config.LOG_DIR.ckpt, config.mode, config.max_ckpt_num)
    state, ckpt_name = ckpt_manager.load_ckpt(network, by_score = config.EVAL.load_ckpt_by_score, name = config.EVAL.ckpt_name, abs_name = config.EVAL.ckpt_abs_name, epoch = config.EVAL.ckpt_epoch)
    print('EVAL', state, ckpt_name)

    save_path_root = config.EVAL.LOG_DIR.save

    save_path_root_deblur = os.path.join(save_path_root, mode, ckpt_name.split('.')[0])
    save_path_root_deblur_score = save_path_root_deblur
    Path(save_path_root_deblur).mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), os.path.join(save_path_root_deblur, ckpt_name))
    save_path_root_deblur = os.path.join(save_path_root_deblur, config.EVAL.data, date)
    Path(save_path_root_deblur).mkdir(parents=True, exist_ok=True)

    config.EVAL.c_path = None
    config.EVAL.l_path = None
    config.EVAL.r_path = None

    if config.EVAL.data == 'DPDD':
        config.EVAL.c_path = '/data1/junyonglee/dd_dp_dataset_canon/dd_dp_dataset_png/test_c'
        config.EVAL.l_path = '/data1/junyonglee/dd_dp_dataset_canon/dd_dp_dataset_png/test_l'
        config.EVAL.r_path = '/data1/junyonglee/dd_dp_dataset_canon/dd_dp_dataset_png/test_r'
    elif config.EVAL.data == 'pixel':
        config.EVAL.c_path = '/data1/junyonglee/dd_dp_dataset_canon/dd_dp_dataset_pixel/test_c'
        config.EVAL.l_path = '/data1/junyonglee/dd_dp_dataset_canon/dd_dp_dataset_pixel/test_l'
        config.EVAL.r_path = '/data1/junyonglee/dd_dp_dataset_canon/dd_dp_dataset_pixel/test_r'
    elif config.EVAL.data == 'RealDOF':
        config.EVAL.c_path = '/data1/junyonglee/dd_dp_dataset_canon/RealDOF_testset/test_c'
    elif config.EVAL.data == 'CUHK':
        config.EVAL.c_path = '/data1/junyonglee/dd_dp_dataset_canon/CUHK/test_c'

    input_c_folder_path_list, input_c_file_path_list, _ = load_file_list(config.EVAL.c_path, config.EVAL.input_path)
    if config.EVAL.l_path is not None:
        input_l_folder_path_list, input_l_file_path_list, _ = load_file_list(config.EVAL.l_path, config.EVAL.input_path)
        input_r_folder_path_list, input_r_file_path_list, _ = load_file_list(config.EVAL.r_path, config.EVAL.input_path)
    else:
        input_l_folder_path_list = input_c_folder_path_list
        input_l_file_path_list = input_c_file_path_list
        input_r_folder_path_list = input_c_folder_path_list
        input_r_file_path_list = input_c_file_path_list

    gt_folder_path_list, gt_file_path_list, _ = load_file_list(config.EVAL.c_path, config.EVAL.gt_path)

    save_path_deblur = os.path.join(save_path_root_deblur)
    Path(save_path_deblur).mkdir(parents=True, exist_ok=True)

    return network, save_path_deblur, save_path_root_deblur_score, ckpt_name, input_c_file_path_list, input_l_file_path_list, input_r_file_path_list, gt_file_path_list

def eval_quan(config):
    mode = 'quantitative'
    network, save_path_deblur, save_path_root_deblur_score, ckpt_name, input_c_file_path_list, input_l_file_path_list, input_r_file_path_list, gt_file_path_list = init(config, mode)

    ##
    time_norm = 0
    total_itr_time = 0
    PSNR_mean = 0.
    SSIM_mean = 0.
    MAE_mean = 0.
    LPIPS_mean = 0.
    LPIPSN = LPIPS.PerceptualLoss(model='net-lin',net='alex').to(torch.device('cuda'))
    ##
    for i, frame_name in enumerate(input_c_file_path_list):
        refine_val = 16

        if config.EVAL.data == 'pixel':
            rotate = cv2.ROTATE_90_COUNTERCLOCKWISE
        else:
            rotate = None

        L = refine_image(read_frame(input_l_file_path_list[i], config.norm_val, rotate), refine_val)
        L = torch.FloatTensor(L.transpose(0, 3, 1, 2).copy()).cuda()

        R = refine_image(read_frame(input_r_file_path_list[i], config.norm_val, rotate), refine_val)
        R = torch.FloatTensor(R.transpose(0, 3, 1, 2).copy()).cuda()

        C = refine_image(read_frame(input_c_file_path_list[i], config.norm_val, rotate), refine_val)
        C = torch.FloatTensor(C.transpose(0, 3, 1, 2).copy()).cuda()

        GT = refine_image(read_frame(gt_file_path_list[i], config.norm_val, rotate), refine_val)
        GT = torch.FloatTensor(GT.transpose(0, 3, 1, 2).copy()).cuda()

        time_norm = time_norm + 1
        with torch.no_grad():
            if 'dual' not in config.mode:
                init_time = time.time()
                out = network(C, is_train=False)
            else:
                init_time = time.time()
                out = network(C, R, L, is_train=False)
            itr_time = time.time() - init_time
            total_itr_time = total_itr_time + itr_time

            output = out['result']

        output_cpu = output.cpu().numpy()[0].transpose(1, 2, 0) #[0, 1]
        GT_cpu = GT.cpu().numpy()[0].transpose(1, 2, 0) #[0, 1]

        PSNR = psnr(output_cpu, GT_cpu)
        SSIM = ssim(output_cpu, GT_cpu)
        MAE = mae(output_cpu, GT_cpu)
        with torch.no_grad():
            LPIPs = LPIPSN.forward(output * 2. - 1., GT * 2. - 1.).item() #[-1, 1]

        frame_name = os.path.basename(frame_name)
        frame_name, _ = os.path.splitext(frame_name)

        Path(os.path.join(save_path_deblur, 'png')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(save_path_deblur, 'input', 'png')).mkdir(parents=True, exist_ok=True) # for CUHK

        save_file_path_deblur = os.path.join(save_path_deblur, 'png', '{:02d}.png'.format(i+1))
        save_file_path_deblur_input = os.path.join(save_path_deblur, 'input', 'png', '{:02d}.png'.format(i+1)) # for CUHK

        if config.EVAL.data == 'pixel':
            output = torch.rot90(output, 1, [3, 2])

        vutils.save_image(output, '{}'.format(save_file_path_deblur), nrow=1, padding = 0, normalize = False)
        vutils.save_image(C, '{}'.format(save_file_path_deblur_input), nrow=1, padding = 0, normalize = False) # for cuhk

        Path(os.path.join(save_path_deblur, 'jpg')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(save_path_deblur, 'input', 'jpg')).mkdir(parents=True, exist_ok=True) # for CUHK
        save_file_path_deblur = os.path.join(save_path_deblur, 'jpg', '{:02d}.jpg'.format(i+1))
        save_file_path_deblur_input = os.path.join(save_path_deblur, 'input', 'jpg', '{:02d}.jpg'.format(i+1)) # for CUHK

        vutils.save_image(output, '{}'.format(save_file_path_deblur), nrow=1, padding = 0, normalize = False)
        vutils.save_image(C, '{}'.format(save_file_path_deblur_input), nrow=1, padding = 0, normalize = False) # for cuhk

        print('[EVAL {}|{}][{}/{}] {} PSNR: {:.5f}, SSIM: {:.5f}, MAE: {:.5f}, LPIPS: {:.5f} ({:.5f}sec)'.format(config.mode, config.EVAL.data, i + 1, len(input_c_file_path_list), frame_name, PSNR, SSIM, MAE, LPIPs, itr_time))
        with open(os.path.join(save_path_root_deblur_score, 'score_{}.txt'.format(config.EVAL.data)), 'w' if i == 0 else 'a') as file:
            file.write('[EVAL {}][{}/{}] {} PSNR: {:.5f}, SSIM: {:.5f}, MAE: {:.5f}, LPIPS: {:.5f} ({:.5f}sec)\n'.format(config.mode, i + 1, len(input_c_file_path_list), frame_name, PSNR, SSIM, MAE, LPIPs, itr_time))
            file.close()

        PSNR_mean += PSNR
        SSIM_mean += SSIM
        MAE_mean += MAE
        LPIPS_mean += LPIPs

        gc.collect()

    total_itr_time = total_itr_time / time_norm

    PSNR_mean = PSNR_mean / len(input_c_file_path_list)
    SSIM_mean = SSIM_mean / len(input_c_file_path_list)
    MAE_mean = MAE_mean / len(input_c_file_path_list)
    LPIPS_mean = LPIPS_mean / len(input_c_file_path_list)

    sys.stdout.write('\n[TOTAL {}|{}] PSNR: {:.5f} SSIM: {:.5f} MAE: {:.5f} LPIPS: {:.5f} ({:.5f}sec)'.format(ckpt_name, config.EVAL.data, PSNR_mean, SSIM_mean, MAE_mean, LPIPS_mean, total_itr_time))
    with open(os.path.join(save_path_root_deblur_score, 'score_{}.txt'.format(config.EVAL.data)), 'a') as file:
        file.write('\n[TOTAL {}] PSNR: {:.5f} SSIM: {:.5f} MAE: {:.5f} LPIPS: {:.5f} ({:.5f}sec)'.format(ckpt_name, PSNR_mean, SSIM_mean, MAE_mean, LPIPS_mean, total_itr_time))
        file.close()

def eval(config):
    if config.EVAL.mode == 'quan':
        eval_quan(config)
