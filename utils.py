import numpy as np
import operator
import collections
import os
import fnmatch
import termcolor
import time
import string
from shutil import rmtree
from pathlib import Path
import sys

def get_file_path(path, regex):
    file_path = []
    for root, dirnames, filenames in os.walk(path):
        for i in np.arange(len(regex)):
            for filename in fnmatch.filter(filenames, regex[i]):
                file_path.append(os.path.join(root, filename))

    return file_path

def remove_file_end_with(path, regex):
    file_paths = get_file_path(path, [regex])

    for i in np.arange(len(file_paths)):
        os.remove(file_paths[i])

def toRed(content):
    return termcolor.colored(content,"red",attrs=["bold"])

def toGreen(content):
    return termcolor.colored(content,"green",attrs=["bold"])

def toBlue(content):
    return termcolor.colored(content,"blue",attrs=["bold"])

def toCyan(content):
    return termcolor.colored(content,"cyan",attrs=["bold"])

def toYellow(content):
    return termcolor.colored(content,"yellow",attrs=["bold"])

def toMagenta(content):
    return termcolor.colored(content,"magenta",attrs=["bold"])

def toGrey(content):
    return termcolor.colored(content,"grey",attrs=["bold"])

def toWhite(content):
    return termcolor.colored(content,"white",attrs=["bold"])

def print_logs(train_mode, mode, epoch, max_epoch, time_s, iter = '', iter_total = '', errs = '', coefs = '', lr = None, is_overwrite=True):
    CURSOR_UP_ONE = '\x1b[1A' 
    ERASE_LINE = '\x1b[2K'
    # is_overwrite = True

    err_str = ''
    if errs != '':
        for key, val in errs.items():
            if key == 'total':
                err_str = '{}: '.format(key) + toRed('{:0.3f}'.format(val)) + err_str
            else:
                if coefs == '':
                    err_str += ', {}: '.format(key) + toBlue('{:0.3f}'.format(val))
                elif key in list(coefs.keys()):
                    err_str += ', {}: '.format(key) + toBlue('{:0.3f}(*{:0.3f})'.format(val, coefs[key]))

        err_str = toWhite('*LOSS->') + '[' + err_str + ']'

    iter_str = ''
    if iter != '':
        # if iter == 1:
        #     is_overwrite = False
        iter_str = ' ({}/{})'.format(toCyan('{:04}'.format(iter)), toCyan('{:04}'.format(iter_total)))

    lr_str = ''
    if lr is not None:
        for i, v in enumerate(lr):
            lr_str += ' lr{}: {}'.format(i, toGrey('{:1.2e}'.format(v)))


    sys.stdout.write('{}[{}][{}]{}{}{}{}\n'.format(
            CURSOR_UP_ONE + ERASE_LINE if is_overwrite else '\n\n',
            toWhite(train_mode),
            toYellow(mode),
            toWhite(' {} '.format('EP')) + toCyan('{}/{}'.format(epoch, max_epoch)),
            iter_str,
            lr_str,
            toGreen(' {:5.2f}s'.format(time.time() - time_s)),
            )
        )
    sys.stdout.write('{}\r'.format(
            err_str,
            )
        )

def get_dict_with_list(list_key, list_val, default_val = None):
    is_multi_dim = False
    if type(list_val) == list:
        for val in list_val:
            if type(val) == list:
                is_multi_dim = True
                break

    new_dict = collections.OrderedDict()
    for i in np.arange(len(list_key)):
        # epoch range
        if is_multi_dim:
            if len(list_key) == len(list_val):
                list_temp = list_val[i]
            else:
                list_temp = list_val[0]
            if list_temp[1] == -1:
                new_dict[list_key[i]] = [list_temp[0], default_val]
            else:
                new_dict[list_key[i]] = list_temp
        else:
            if type(list_val) is list and len(list_val) == len(list_key):
                new_dict[list_key[i]] = list_val[i]
            else:
                new_dict[list_key[i]] = list_val

    return new_dict

def dict_operations(dict1, op, operand2):
    ops = {'+': operator.add,
           '-': operator.sub,
           '*': operator.mul,
           '/': operator.truediv
           }
    if op != '=':
        if 'dict' in str(type(dict1)).lower() and type(dict1) == type(operand2):
            return collections.OrderedDict(zip(list(dict1.keys()), [ops[op](dict1[key], operand2[key]) for key in dict1.keys()]))
        elif type(operand2) == list:
            return collections.OrderedDict(zip(list(dict1.keys()), [ops[op](dict1[key], operand2[count]) for count, key in enumerate(dict1.keys())]))
        elif type(operand2) == int or type(operand2) == float:
            return collections.OrderedDict(zip(list(dict1.keys()), [ops[op](dict1[key], operand2) for key in dict1.keys()]))
    else:
        new_dict = collections.OrderedDict()
        for key in dict1.keys():
            new_dict[dict1[key]] = operand2[key]

        return new_dict

def string_to_array(text):
    x = [word.strip(string.punctuation) for word in text.split()]
    return x

def exists_or_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        return True

def handle_directory(config, delete_log):
    def mkdir(dir_dict, delete_log_, delete_ckpt_ = True):
        for (key, val) in dir_dict.items():
            if 'perm' in key and delete_ckpt_ is False:
                exists_or_mkdir(val)
                continue
            if delete_log_:
                rmtree(val, ignore_errors = True)

            exists_or_mkdir(val)

    delete_log = delete_log
    if delete_log:
        delete_log = input('Are you sure to delete the logs (y/n):\n') # python3

        if len(delete_log) == 0 or delete_log[0].lower() == 'y':
            delete_log = True
        elif delete_log[0].lower() == 'n':
            delete_log = False
        else:
            print('invalid input')
            exit()

    mkdir(config.LOG_DIR, delete_log)
