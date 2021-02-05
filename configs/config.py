from easydict import EasyDict as edict
import json
import os
import collections
import numpy as np

def get_config(project = '', mode = '', config_ = ''):
    ## GLOBAL
    config = edict()
    config.project = project
    config.mode = mode
    config.config = config_
    config.is_train = False
    config.thread_num = 1
    config.dist = False
    config.resume = None # 'resume state file name'
    config.manual_seed = 0
    config.is_verbose = False

    ##################################### TRAIN #####################################
    config.trainer = ''
    config.network = ''

    config.in_bit = 8
    config.norm_val = (2**config.in_bit - 1)

    config.batch_size = 8
    config.batch_size_test = 1 #3 for sample recurrent2
    config.height = 256
    config.width = 256

    # learning rate
    config.lr_init = 1e-4
    config.gc = 1.0

    ## Naive Decay
    config.LRS = 'LD' # LD
    config.total_itr = 600000
    config.decay_period = [500000, 550000]
    config.decay_rate = 0.5
    config.warmup_itr = -1

    # adam
    config.beta1 = 0.9

    # data dir
    config.lmdb = False
    config.lmdb_path = '/data1/junyonglee/dd_dp_dataset_canon/lmdb/DP'

    config.c_path = '/data1/junyonglee/dd_dp_dataset_canon/dd_dp_dataset_png/train_c'
    config.l_path = '/data1/junyonglee/dd_dp_dataset_canon/dd_dp_dataset_png/train_l'
    config.r_path = '/data1/junyonglee/dd_dp_dataset_canon/dd_dp_dataset_png/train_r'
    config.d_path = '/data1/junyonglee/dd_dp_dataset_canon/DMENet/train_c'
    config.m_path = '/data1/junyonglee/dd_dp_dataset_canon/dd_dp_dataset_png/train_c/sharp_mask_target'

    config.input_path = 'source'
    config.gt_path = 'target'

    # logs
    config.max_ckpt_num = 100
    config.write_ckpt_every_epoch = 4
    config.refresh_image_log_every_epoch = {'train':20, 'valid':20}
    config.write_log_every_itr = {'train':200, 'valid': 1}

    # log dirs
    config.LOG_DIR = edict()
    offset = './logs'
    offset = os.path.join(offset, config.project)
    offset = os.path.join(offset, '{}'.format(mode))
    config.LOG_DIR.offset = offset
    config.LOG_DIR.ckpt = os.path.join(config.LOG_DIR.offset, 'checkpoint', 'train', 'epoch')
    config.LOG_DIR.ckpt_ckpt = os.path.join(config.LOG_DIR.offset, 'checkpoint', 'train', 'epoch', 'ckpt')
    config.LOG_DIR.ckpt_state = os.path.join(config.LOG_DIR.offset, 'checkpoint', 'train', 'epoch', 'state')
    config.LOG_DIR.log_scalar = os.path.join(config.LOG_DIR.offset, 'log', 'train', 'scalar')
    config.LOG_DIR.log_image = os.path.join(config.LOG_DIR.offset, 'log', 'train', 'image', 'train')
    config.LOG_DIR.sample = os.path.join(config.LOG_DIR.offset, 'sample', 'train')
    config.LOG_DIR.sample_val = os.path.join(config.LOG_DIR.offset, 'sample', 'valid')
    config.LOG_DIR.config = os.path.join(config.LOG_DIR.offset, 'config')

    ################################## VALIDATION ###################################
    # data path
    config.VAL = edict()
    config.VAL.val_offset = 'test'
    # config.VAL.val_offset = '/data1/junyonglee'
    config.VAL.c_path = os.path.join(config.VAL.val_offset, 'dd_dp_dataset_png/val_c')
    config.VAL.l_path = os.path.join(config.VAL.val_offset, 'dd_dp_dataset_png/val_l')
    config.VAL.r_path = os.path.join(config.VAL.val_offset, 'dd_dp_dataset_png/val_r')
    config.VAL.input_path = 'source' # os.path.join(config.VAL.data_path, 'input')
    config.VAL.gt_path = 'target' # os.path.join(config.VAL.data_path, 'gt')

    ##################################### EVAL ######################################
    config.EVAL = edict()
    config.EVAL.eval_mode = 'quan'
    config.EVAL.data = 'DPDD' # DPDD/PixelDP/RealDOF/CUHK

    config.EVAL.load_ckpt_by_score = True
    config.EVAL.ckpt_name = None
    config.EVAL.ckpt_epoch = None
    config.EVAL.ckpt_abs_name = None
    config.EVAL.low_res = False
    config.EVAL.ckpt_load_path = None

    # data dir
    config.EVAL.test_offset = 'test'
    # config.EVAL.test_offset = '/data1/junyonglee'
    config.EVAL.c_path = None
    config.EVAL.l_path = None
    config.EVAL.r_path = None

    config.EVAL.input_path = 'source' # os.path.join(offset, 'input')
    config.EVAL.gt_path = 'target' # os.path.join(offset, 'gt')

    # log dir
    config.EVAL.LOG_DIR = edict()
    config.EVAL.LOG_DIR.save = os.path.join(config.LOG_DIR.offset, 'result')

    return config

def set_eval_path(config, data):
    if data == 'DPDD':
        config.EVAL.c_path = os.path.join(config.EVAL.test_offset, 'DPDD/test_c')
        config.EVAL.l_path = os.path.join(config.EVAL.test_offset, 'DPDD/test_l')
        config.EVAL.r_path = os.path.join(config.EVAL.test_offset, 'DPDD/test_r')
    elif data == 'PixelDP':
        config.EVAL.c_path = os.path.join(config.EVAL.test_offset, 'PixelDP/test_c')
        config.EVAL.l_path = os.path.join(config.EVAL.test_offset, 'PixelDP/test_l')
        config.EVAL.r_path = os.path.join(config.EVAL.test_offset, 'PixelDP/test_r')
    elif data == 'RealDOF':
        config.EVAL.c_path = os.path.join(config.EVAL.test_offset, 'RealDOF/test_c')
    elif data == 'CUHK':
        config.EVAL.c_path = os.path.join(config.EVAL.test_offset, 'CUHK/test_c')

    return config

def log_config(path, cfg):
    with open(path + '/config.txt', 'w') as f:
        f.write(json.dumps(cfg, indent=4))
        f.close()


def print_config(cfg):
    print(json.dumps(cfg, indent=4))

