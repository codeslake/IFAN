import collections
import numpy as np
import cv2
from threading import Thread
from threading import Lock
from datetime import datetime
import torch
from data_loader.utils import *

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

class Data_Loader:
    def __init__(self, config, is_train, name, thread_num = 3):

        self.name = name
        self.is_train = is_train
        self.thread_num = thread_num

        if is_train:
            self.input_folder_path_list, self.input_file_path_list, _ = load_file_list(config.data_path, config.input_path)
            self.gt_folder_path_list, self.gt_file_path_list, _ = load_file_list(config.data_path, config.gt_path)
            self.frame_num = config.frame_num
            self.batch_size = config.batch_size
        else:
            self.input_folder_path_list, self.input_file_path_list, _ = load_file_list(config.VAL.data_path, config.VAL.input_path)
            self.gt_folder_path_list, self.gt_file_path_list, _ = load_file_list(config.VAL.data_path, config.VAL.gt_path)
            self.frame_num = config.frame_num_test
            self.batch_size = config.batch_size_test

        self.h = config.height
        self.w = config.width

        self.is_color = config.is_color
        self.is_augment = config.is_augment
        self.is_reverse = config.is_augment

    def init_data_loader(self, inputs):

        self.idx_video = []
        self.idx_frame = []
        self._init_idx()

        self.num_itr = int(np.ceil(len(sum(self.idx_frame, [])) / self.batch_size))

        self.lock = Lock()
        self.is_end = False

        ### THREAD HOLDERS ###
        self.net_placeholder_names = list(inputs.keys())
        self.net_inputs = inputs
        self.threads = [None] * self.thread_num
        self.need_to_be_used = [None] * self.thread_num
        self.feed_dict_holder = get_dict_array_by_key(self.net_placeholder_names, self.thread_num)
        self._init_data_thread()

    def _init_idx(self):
        self.idx_video = []
        self.idx_frame = []
        for i in range(len(self.input_file_path_list)):
            total_frames = len(self.input_file_path_list[i])

            idx_frame_temp = list(range(0, total_frames - self.frame_num + 1))

            self.idx_frame.append(idx_frame_temp)
            self.idx_video.append(i)

        self.is_end = False

    def get_feed(self):
        thread_idx, is_end = self._get_thread_idx()
        #tl.logging.debug('[%s] THREAD[%s] > FEED_THE_NETWORK [%s]' % (self.name, str(thread_idx), str(datetime.now())))
        if is_end:
            return None, is_end

        # feed_dict = collections.OrderedDict()
        for (key, val) in self.net_inputs.items():
            self.net_inputs[key] = self.feed_dict_holder[key][thread_idx]

        #tl.logging.debug('[%s] THREAD[%s] > FEED_THE_NETWORK DONE [%s]' % (self.name, str(thread_idx), str(datetime.now())))
        return self.net_inputs, is_end

    def _get_batch(self, need_to_be_used, thread_idx):
        assert(self.net_placeholder_names is not None)
        # print('[%s] \tthread[%s] > _get_batch start [%s]' % (self.name, str(thread_idx), str(datetime.now())))

        ## random sample frame indexes
        self.lock.acquire()
        # print('[%s] \t\tthread[%s] > acquired lock [%s]' % (self.name, str(thread_idx), str(datetime.now())))

        if self.is_end:
            # print('[%s] \t\tthread[%s] > releasing lock 1 [%s]' % (self.name, str(thread_idx), str(datetime.now())))
            self.lock.release()
            return

        video_idxes = []
        frame_offsets = []

        actual_batch = 0
        for i in range(0, self.batch_size):
            if len(self.idx_video) == 0:
                self.is_end = True
                # tl.logging.debug('[%s] \t\tthread[%s] > releasing lock 2 [%s]' % (self.name, str(thread_idx), str(datetime.now())))
                self.lock.release()
                return
            else:
                if self.is_train:
                    idx_x = np.random.randint(len(self.idx_video))
                    video_idx = self.idx_video[idx_x]
                    idx_y = np.random.randint(len(self.idx_frame[video_idx]))
                else:
                    idx_x = 0
                    idx_y = 0
                    video_idx = self.idx_video[idx_x]

            frame_offset = self.idx_frame[video_idx][idx_y]
            video_idxes.append(video_idx)
            frame_offsets.append(frame_offset)
            self._update_idx(idx_x, idx_y)
            actual_batch += 1

        # print('[%s] \t\tthread[%s] > releasing lock 4 [%s]' % (self.name, str(thread_idx), str(datetime.now())))
        #if self.is_train is False:
            #print('idx: ', thread_idx, ' / is_end: ', self.is_end)
        self.lock.release()
        need_to_be_used[thread_idx] = True

        ## init thread lists
        data_holder = get_dict_array_by_key(self.net_placeholder_names, actual_batch)

        ## start thread
        threads = [None] * actual_batch
        for batch_idx in range(actual_batch):
            video_idx = video_idxes[batch_idx]
            frame_offset = frame_offsets[batch_idx]
            is_reverse = np.random.randint(2) if self.is_reverse else 0
            threads[batch_idx] = Thread(target = self._read_dataset, args = (data_holder, batch_idx, video_idx, frame_offset, is_reverse))
            threads[batch_idx].start()

        for batch_idx in range(actual_batch):
            threads[batch_idx].join()

        for (key, val) in data_holder.items():
            data_holder[key] = np.concatenate(data_holder[key][0 : actual_batch], axis = 0)

        for holder_name in self.net_placeholder_names:
            self.feed_dict_holder[holder_name][thread_idx] = torch.FloatTensor(data_holder[holder_name]).to(torch.device('cuda'))

        # print('[%s] \tthread[%s] > _get_batch done [%s]' % (self.name, str(thread_idx), str(datetime.now())))

    def _read_dataset(self, data_holder, batch_idx, video_idx, frame_offset, is_reverse):
        sampled_frame_idx = np.arange(frame_offset, frame_offset + self.frame_num)
        if is_reverse:
            sampled_frame_idx = np.flip(sampled_frame_idx)

        input_patches_temp = [None] * len(sampled_frame_idx)
        gt_patches_temp = [None] * len(sampled_frame_idx)

        threads = [None] * len(sampled_frame_idx)
        for frame_idx in range(len(sampled_frame_idx)):
            sampled_idx = sampled_frame_idx[frame_idx]

            threads[frame_idx] = Thread(target = self._read_data, args = (data_holder, batch_idx, video_idx, frame_idx, sampled_idx, input_patches_temp, gt_patches_temp))
            threads[frame_idx].start()

        for frame_idx in range(len(sampled_frame_idx)):
            threads[frame_idx].join()

        input_patches_temp = np.concatenate(input_patches_temp[:len(sampled_frame_idx)], axis = 3)
        gt_patches_temp = np.concatenate(gt_patches_temp[:len(sampled_frame_idx)], axis = 3)

        cropped_frames = np.concatenate([input_patches_temp, gt_patches_temp], axis = 3)

        if self.is_train:
            cropped_frames = crop_multi(cropped_frames, self.h, self.w, is_random = True)
        else:
            cropped_frames = crop_multi(cropped_frames, self.h, self.w, is_random = False)

        input_patches = cropped_frames[:, :, :, 0:len(sampled_frame_idx) * 3]
        input_patches = input_patches.reshape((1, self.h, self.w, -1, 3))
        input_patches = np.transpose(input_patches, (0, 3, 1, 2, 4))

        gt_patches = cropped_frames[:, :, :, len(sampled_frame_idx) * 3:]
        gt_patches = gt_patches.reshape((1, self.h, self.w, -1, 3))
        gt_patches = np.transpose(gt_patches, (0, 3, 1, 2, 4))
        # gt_patches = gt_patches[:, len(sampled_frame_idx) // 2, :, :, :]

        data_holder['input'][batch_idx] = input_patches.transpose([0, 1, 4, 2, 3])
        data_holder['gt'][batch_idx] = gt_patches.transpose([0, 1, 4, 2, 3])

    def _read_data(self, data_holder, batch_idx, video_idx, frame_idx, sampled_idx, input_patches_temp, gt_patches_temp):
        # read stab frame
        input_file_path = self.input_file_path_list[video_idx]
        gt_file_path = self.gt_file_path_list[video_idx]

        # print(get_folder_name(str(Path(input_file_path[sampled_idx]).parent)), get_folder_name(str(Path(gt_file_path[sampled_idx]).parent)))
        # print(get_base_name(input_file_path[sampled_idx]), get_base_name(gt_file_path[sampled_idx]))
        assert(get_folder_name(str(Path(input_file_path[sampled_idx]).parent)) == get_folder_name(str(Path(gt_file_path[sampled_idx]).parent)))
        assert(get_base_name(input_file_path[sampled_idx]) == get_base_name(gt_file_path[sampled_idx]))

        input_frame = read_frame(input_file_path[sampled_idx])
        gt_frame = read_frame(gt_file_path[sampled_idx])
        # print(self.name, input_frame.shape)
        # print(self.name, gt_frame.shape)

        input_patches_temp[frame_idx] = input_frame
        gt_patches_temp[frame_idx] = gt_frame

    def _update_idx(self, idx_x, idx_y):
        video_idx = self.idx_video[idx_x]
        del(self.idx_frame[video_idx][idx_y])

        if len(self.idx_frame[video_idx]) == 0:
            del(self.idx_video[idx_x])

    def _init_data_thread(self):
        self._init_idx()
        #tl.logging.debug('[%s] INIT_THREAD [%s]' % str(self.name, datetime.now()))
        for thread_idx in range(0, self.thread_num):
            self.threads[thread_idx] = Thread(target = self._get_batch, args = (self.need_to_be_used, thread_idx))
            self.need_to_be_used[thread_idx] = False
            self.threads[thread_idx].start()
        #tl.logging.debug('[%s] INIT_THREAD DONE [%s]' % str(self.name, datetime.now()))

    def _get_thread_idx(self):
        while True:
            # if there is resting thread that has a thing to return, return it and make the thread to work again unless all the data is returned
            for thread_idx in np.arange(self.thread_num):
                if self.need_to_be_used[thread_idx] and self.threads[thread_idx].is_alive() == False:
                    self.need_to_be_used[thread_idx] = False

                    if self.is_end == False:
                        self.threads[thread_idx].join()
                        self.threads[thread_idx] = Thread(target = self._get_batch, args = (self.need_to_be_used, thread_idx))
                        self.threads[thread_idx].start()

                    return thread_idx, False

            # if all threads have nothing to return and all of them are resting, notify the trainer that it is end of dataset
            all_done = True
            for thread_idx in np.arange(self.thread_num):
                if self.need_to_be_used[thread_idx] or self.threads[thread_idx].is_alive():
                    all_done = False

            if all_done and self.is_end:
                self._init_data_thread()
                return None, True

