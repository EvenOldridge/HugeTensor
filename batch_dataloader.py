import torch
from torch import _utils

class BatchDataLoader(object):
    __initialized = False

    def __init__(self, batchdataset, shuffle=False,
                 pin_memory=False, num_workers = 0):
        self.batchdataset = batchdataset
        self.batch_size = batchdataset.batch_size
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.__initialized = True

    def __setattr__(self, attr, val):
        if self.__initialized and attr in ('batch_size', 'sampler', 'drop_last'):
            raise ValueError('{} attribute should not be set after {} is '
                             'initialized'.format(attr, self.__class__.__name__))

        super(BatchDataLoader, self).__setattr__(attr, val)

    def __iter__(self):
        return _BatchDataLoaderIter(self)

    def __len__(self):
        return len(self.batchdataset) // self.batch_size

    
class _BatchDataLoaderIter(object):
    def __init__(self, loader):
        self.batchdataset = loader.batchdataset
        self.batch_size = loader.batch_size
        self.pin_memory = loader.pin_memory and torch.cuda.is_available()

        if loader.shuffle:
            self.batchdataset.shuffle()
        self.idx = 0

    def __len__(self):
        return len(self.batchdataset) // self.batch_size
    
    def __next__(self):
        if self.idx+1 > len(self):
            raise StopIteration
        batch = self.batchdataset[self.idx]
        # Note Pinning memory was ~10% _slower_ for the test examples I explored
        if self.pin_memory:
            batch = _utils.pin_memory.pin_memory_batch(batch)
        self.idx = self.idx+1
        return batch

    next = __next__  # Python 2 compatibility

    def __iter__(self):
        return self
