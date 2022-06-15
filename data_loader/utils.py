import os
import numpy as np
import cv2
from pathlib import Path
import collections

def read_frame(path, norm_val = None, rotate = None):
    if norm_val == (2**16-1):
        frame = cv2.imread(path, -1)
        if rotate is not None:
            frame = cv2.rotate(frame, rotate)
        frame = frame / norm_val
        frame = frame[...,::-1]
    else:
        frame = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if rotate is not None:
            frame = cv2.rotate(frame, rotate)
        frame = frame / 255.


    return np.expand_dims(frame, axis = 0)

def crop_multi(x, hrg, wrg, is_random=False, row_index=0, col_index=1):

    h, w = x[0].shape[row_index], x[0].shape[col_index]

    if (h <= hrg) or (w <= wrg):
        raise AssertionError("The size of cropping should smaller than the original image")

    if is_random:
        h_offset = int(np.random.uniform(0, h - hrg) - 1)
        w_offset = int(np.random.uniform(0, w - wrg) - 1)
        results = []
        for data in x:
            results.append(data[int(h_offset):int(hrg + h_offset), int(w_offset):int(wrg + w_offset)])
        return np.asarray(results)
    else:
        # central crop
        h_offset = (h - hrg) / 2
        w_offset = (w - wrg) / 2
        results = []
        for data in x:
            results.append(data[int(h_offset):int(h - h_offset), int(w_offset):int(w - w_offset)])
        return np.asarray(results)

def color_to_gray(img):
    c_linear = 0.2126*img[:, :, 0] + 0.7152*img[:, :, 1] + 0.07228*img[:, :, 2]
    c_linear_temp = c_linear.copy()

    c_linear_temp[np.where(c_linear <= 0.0031308)] = 12.92 * c_linear[np.where(c_linear <= 0.0031308)]
    c_linear_temp[np.where(c_linear > 0.0031308)] = 1.055 * np.power(c_linear[np.where(c_linear > 0.0031308)], 1.0/2.4) - 0.055

    img[:, :, 0] = c_linear_temp
    img[:, :, 1] = c_linear_temp
    img[:, :, 2] = c_linear_temp

    return img

def refine_image(img, val = 16):
    shape = img.shape
    if len(shape) == 4:
        _, h, w, _ = shape[:]
        return img[:, 0 : h - h % val, 0 : w - w % val, :]
    elif len(shape) == 3:
        h, w = shape[:2]
        return img[0 : h - h % val, 0 : w - w % val, :]
    elif len(shape) == 2:
        h, w = shape[:2]
        return img[0 : h - h % val, 0 : w - w % val]

def refine_image_pt(image, val = 16):
    size = image.size()
    h = size[2]
    w = size[3]
    refine_h = h - h % val
    refine_w = w - w % val

    return image[:, :, :refine_h, :refine_w]

def load_file_list(root_path, child_path = None, is_flatten = False):
    folder_paths = []
    filenames_pure = []
    filenames_structured = []
    num_files = 0
    for root, dirnames, filenames in os.walk(root_path):
        # print('Root:', root, ', dirnames:', dirnames, ', filenames', filenames)
        if len(dirnames) != 0:
            if dirnames[0][0] == '@':
                del(dirnames[0])

        if len(dirnames) == 0:
            if root == '.':
                continue
            if child_path is not None and child_path != Path(root).name:
                continue
            folder_paths.append(root)
            filenames_pure = []
            for i in np.arange(len(filenames)):
                if filenames[i][0] != '.' and filenames[i] != 'Thumbs.db':
                    filenames_pure.append(os.path.join(root, filenames[i]))
            # print('filenames_pure:', filenames_pure)
            filenames_structured.append(np.array(sorted(filenames_pure), dtype='str'))
            num_files += len(filenames_pure)

    folder_paths = np.array(folder_paths)
    filenames_structured = np.array(filenames_structured, dtype=object)

    sort_idx = np.argsort(folder_paths)
    folder_paths = folder_paths[sort_idx]
    filenames_structured = filenames_structured[sort_idx]

    if is_flatten:
        if len(filenames_structured) > 1:
            filenames_structured = np.concatenate(filenames_structured).ravel()
        else:
            filenames_structured = filenames_structured.flatten()

    return folder_paths, filenames_structured, num_files

def get_base_name(path):
    return os.path.basename(path.split('.')[0])

def get_folder_name(path):
    path = os.path.dirname(path)
    return path.split(os.sep)[-1]

