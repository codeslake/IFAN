"""
Modified from torch.utils.data.distributed.DistributedSampler
Support enlarging the dataset for *iteration-oriented* training, for saving time when restart the
dataloader after each epoch
"""
import math
import torch
from torch.utils.data.sampler import Sampler
import torch.distributed as dist


class DistIterSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, ratio=1, is_train = True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.is_train = is_train
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * ratio / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):

        if self.is_train:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(int(self.epoch))
            indices = torch.randperm(self.total_size, generator=g).tolist()

            dsize = len(self.dataset)
            indices = [v % dsize for v in indices]

            # subsample
            indices = indices[self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples

            return iter(indices)

        else:
            # deterministically shuffle based on epoch
            dsize = len(self.dataset)
            indices = list(range(dsize))

            indices = [v % dsize for v in indices]

            # subsample
            indices = indices[self.rank:len(self.dataset):self.num_replicas]
            # assert len(indices) == self.num_samples

            return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
