from configs.config import get_config as main_config
from configs.config import log_config, print_config

def get_config(project = '', mode = '', config = ''):
    ## GLOBAL
    config = main_config(project, mode, config)

    ## LOCAL
    config.trainer = 'trainer'
    config.is_amp = True
    config.ks = 3
    config.ch = 32
    config.res_num = 2
    config.in_bit = 8
    config.norm_val = (2**config.in_bit - 1)

    config.Fs = 3 # filter size
    config.N = 17
    config.refine_val = 8

    # weight init
    config.wi = 1.0
    config.wiF = 1.5
    config.wiRF = 1.0

    # for reblurring
    config.RBFs = 3
    config.RBF_num = 17

    # gradient clipping
    config.gc = 0.5

    # max_sigma for Gaussian noise
    config.max_sig = 0.005**(1/2) # 0.07

    return config

