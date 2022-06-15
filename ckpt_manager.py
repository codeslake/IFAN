import os
from shutil import *

import torch
import numpy as np

class CKPT_Manager:
    def __init__(self, root_dir, model_name, cuda, max_files_to_keep = 10, is_descending = False):
        self.root_dir = root_dir
        self.root_dir_ckpt = os.path.join(root_dir, 'ckpt')
        self.root_dir_state = os.path.join(root_dir, 'state')
        self.cuda = cuda

        self.model_name = model_name
        self.max_files = max_files_to_keep

        self.ckpt_list = os.path.join(self.root_dir, 'checkpoints.txt')
        self.is_descending = is_descending

    def load_ckpt(self, network, by_score = True, name = None, abs_name = None, epoch = None):
        # get ckpt path
        if name is None and abs_name is None and epoch is None:
            try:
                with open(self.ckpt_list, 'r') as file:
                    lines = file.read().splitlines()
                    file.close()
            except:
                print('ckpt_list does not exists')
                return

            if by_score:
                file_name = lines[0].split(' ')[0]
            else:
                file_name = lines[-1].split(' ')[0]

            file_path = os.path.join(self.root_dir_ckpt, file_name)
        else:
            if name is not None:
                file_name = name
                file_path = os.path.join(self.root_dir_ckpt, file_name)
            if abs_name is not None:
                file_name = os.path.basename(abs_name)
                file_path = abs_name
            if epoch is not None:
                file_name = '{}_{:05d}.pytorch'.format(self.model_name, epoch)
                file_path = os.path.join(self.root_dir_ckpt, file_name)

        if self.cuda is False:
            state_dict = torch.load(file_path, map_location='cpu')
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.split('.', 1)[-1]
                new_state_dict[k] = v
            return network.load_state_dict(new_state_dict, strict=False), os.path.basename(file_name)

        else:
            device_id = torch.cuda.current_device()
            return network.load_state_dict(torch.load(file_path, map_location="cuda:{}".format(device_id) if self.cuda else "cpu"), strict=False), os.path.basename(file_name)


    def resume(self, network, resume_name=None, resume_abs=None, rank = -1):
        # todo
        # according to the resume_name,

        # 2. load ckpt
        if resume_name is not None:
            resume_name = self.model_name + '_' + '{:05d}'.format(int(resume_name)) + '.pytorch'
            ckpt_dir = os.path.join(self.root_dir_ckpt, resume_name)
            state_dir = os.path.join(self.root_dir_state, resume_name)
        elif resume_abs is not None:
            ckpt_dir = resume_abs
            # path = Path(resume_abs)
            # root_dir_state = path.parent.parent.absolute()
            # state_dir = os.path.join(root_dir_state, 'state', path.name)
            result, _ = self.load_ckpt(network, abs_name=ckpt_dir)
            print('Rank[{}] resume ckpt loading: '.format(rank), result)
            return None

        result, _ = self.load_ckpt(network, name=ckpt_dir)
        print('Rank[{}] resume ckpt loading: '.format(rank), result)

        # 3. load state
        device_id = torch.cuda.current_device()
        file_name = state_dir
        # resume_state = torch.load(file_name, map_location=lambda storage, loc: storage.cuda(device_id))
        resume_state = torch.load(file_name, map_location="cuda:{}".format(device_id))

        # 4. files in chekpoints?
        # if the resume state is the last file, good to go
        # if the resume state is in the middle of states/checkpoints, remove the states/checkpoints after the resume state
        if rank <= 0:
            with open(self.ckpt_list, 'r') as file:
                lines = file.read().splitlines()
                file.close()

            lines_to_add = []
            line_recent = None
            epoch_to_resume = int(resume_name.split('.')[0].split('_')[-1])
            for line in lines[:-1]:
                file_name = line.split(' ')[0]
                epoch = int(file_name.split('.')[0].split('_')[-1])
                if epoch > epoch_to_resume:
                    try:
                        os.remove(ckpt_dir)
                        os.remove(state_dir)
                    except:
                        pass
                elif epoch == epoch_to_resume:
                    line_recent = line
                    lines_to_add.append(line)
                else:
                    lines_to_add.append(line)

            if line_recent == None:
                line_recent = lines[-1]
            lines_to_add.append(line_recent)

            with open(self.ckpt_list, 'w') as file:
                for line in lines_to_add:
                    file.write(line + os.linesep)
                file.close()

            self._update_files()

        return resume_state

    def save(self, network, state, epoch, score):
        if type(epoch) == str:
            file_name = self.model_name + '_' + epoch + '.pytorch'
        else:
            file_name = self.model_name + '_' + '{:05d}'.format(epoch) + '.pytorch'
        save_path = os.path.join(self.root_dir_ckpt, file_name)
        torch.save(network.state_dict(), save_path)

        save_path = os.path.join(self.root_dir_state, file_name)
        torch.save(state, save_path)

        # remove the most recently added line
        if os.path.exists(self.ckpt_list):
            with open(self.ckpt_list, 'r') as file:
                lines = file.read().splitlines()
                line_to_remove  = lines[-1]
                if line_to_remove not in lines[:-1]:
                    os.remove(os.path.join(self.root_dir_ckpt, line_to_remove.split(' ')[0]))
                    os.remove(os.path.join(self.root_dir_state, line_to_remove.split(' ')[0]))
                del(lines[-1])
                file.close()
            with open(self.ckpt_list, 'w') as file:
                for line in lines:
                    file.write(line + os.linesep)
                file.close()

        with open(self.ckpt_list, 'a') as file:
            #line_to_add = file_name + ' ' + str(score) + os.linesep
            line_to_add = file_name
            for s in score:
                line_to_add = line_to_add + ' ' + str(s)
            line_to_add = line_to_add + os.linesep

            file.write(line_to_add) # for the new ckpt
            file.write(line_to_add) # for the most recent ckpt
            file.close()

        self._update_files()

    def _update_files(self):
        # read file
        with open(self.ckpt_list, 'r') as file:
            lines = file.read().splitlines()
            file.close()

        # sort by score
        line_recent = lines[-1]
        lines_prev = self._sort(lines[:-1])

        # delete ckpt
        while len(lines_prev) > self.max_files:
            line_to_remove = lines_prev[-1]
            if line_to_remove != line_recent:
                os.remove(os.path.join(self.root_dir_ckpt, line_to_remove.split(' ')[0]))
                os.remove(os.path.join(self.root_dir_state, line_to_remove.split(' ')[0]))
            del(lines_prev[-1])

        # update ckpt list
        with open(self.ckpt_list, 'w') as file:
            for line in lines_prev:
                file.write(line + os.linesep)
            file.write(line_recent + os.linesep)
            file.close()

    def _sort(self, lines):
        scores = [float(score.split(' ')[1]) for score in lines]
        lines = [line for _, line in sorted(zip(scores, lines), reverse=self.is_descending)]

        return lines

