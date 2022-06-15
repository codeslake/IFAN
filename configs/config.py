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
    config.thread_num = 8
    config.resume = None # 'resume epoch'
    config.resume_abs = None # 'resume abs name'
    config.manual_seed = 0
    config.is_verbose = False
    config.save_sample = False
    config.is_amp = False

    config.cuda = True
    if config.cuda == True:
        config.device = 'cuda'
    else:
        config.device = 'cpu'
    config.dist = False

    ##################################### TRAIN #####################################
    config.trainer = ''
    config.network = ''

    config.in_bit = 8
    config.norm_val = (2**config.in_bit - 1)

    config.batch_size = 8
    config.batch_size_test = 1
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
    config.data_offset = '/data1/junyonglee/defocus_deblur'
    #config.data_offset = 'datasets/defocus_deblur'
    config.c_path = os.path.join(config.data_offset, 'DPDD/train_c')
    config.l_path = os.path.join(config.data_offset, 'DPDD/train_l')
    config.r_path = os.path.join(config.data_offset, 'DPDD/train_r')

    config.input_path = 'source'
    config.gt_path = 'target'

    # logs
    config.max_ckpt_num = 100
    config.write_ckpt_every_epoch = 4
    config.refresh_image_log_every_epoch = {'train':20, 'valid':20}
    config.write_log_every_itr = {'train':200, 'valid': 1}

    # log dirs
    config.LOG_DIR = edict()
    # log_offset = './logs'
    log_offset = '/Bean/logs/junyonglee'
    log_offset = os.path.join(log_offset, config.project)
    log_offset = os.path.join(log_offset, '{}'.format(mode))
    config.LOG_DIR.offset = log_offset
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
    config.VAL.c_path = os.path.join(config.data_offset, 'DPDD/val_c')
    config.VAL.l_path = os.path.join(config.data_offset, 'DPDD/val_l')
    config.VAL.r_path = os.path.join(config.data_offset, 'DPDD/val_r')
    config.VAL.input_path = 'source'
    config.VAL.gt_path = 'target'

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
    config.EVAL.c_path = None
    config.EVAL.l_path = None
    config.EVAL.r_path = None

    config.EVAL.input_path = None
    config.EVAL.gt_path = None

    # log dir
    config.EVAL.LOG_DIR = edict()
    config.output_offset = os.path.join(config.LOG_DIR.offset, 'result')
    config.EVAL.LOG_DIR.save = config.output_offset

    return config

def set_eval_path(config, data):
    if data == 'DPDD':
        config.EVAL.c_path = os.path.join(config.data_offset, 'DPDD/test_c')
        config.EVAL.l_path = os.path.join(config.data_offset, 'DPDD/test_l')
        config.EVAL.r_path = os.path.join(config.data_offset, 'DPDD/test_r')
        # child paths
        config.EVAL.input_path = 'source'
        config.EVAL.gt_path = 'target'

    elif data == 'PixelDP':
        config.EVAL.c_path = os.path.join(config.data_offset, 'PixelDP/test_c')
        config.EVAL.l_path = os.path.join(config.data_offset, 'PixelDP/test_l')
        config.EVAL.r_path = os.path.join(config.data_offset, 'PixelDP/test_r')
    elif data == 'RealDOF':
        config.EVAL.c_path = os.path.join(config.data_offset, 'RealDOF')
        # child paths
        config.EVAL.input_path = 'source'
        config.EVAL.gt_path = 'target'

    elif data == 'CUHK':
        config.EVAL.c_path = os.path.join(config.data_offset, 'CUHK')

    elif data == 'random':
        config.EVAL.c_path = os.path.join(config.data_offset, 'random')

    return config

def log_config(path, cfg):
    with open(path + '/config.txt', 'w') as f:
        f.write(json.dumps(cfg, indent=4))
        f.close()

def print_config(cfg):
    print(json.dumps(cfg, indent=4))

