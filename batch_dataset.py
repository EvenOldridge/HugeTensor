import cudf, torch
from torch.utils import data as torch_data
from torch.utils.dlpack import from_dlpack
import glob, os
import numpy as np
import pyarrow.parquet as pq
import pdb


# Load parquet file during init

def load_tensors_from_parquet_via_cudf(path, target_name='delinquency_12'):
    gdf = cudf.read_parquet(path)
    target = None
    if target_name in gdf.columns:
        target = from_dlpack(gdf[target_name].astype('float32').to_dlpack())
    # hack because we can't cast a whole dataframe
    for col in gdf.columns:
        gdf[col] = gdf[col].astype('int64')
    tensors = from_dlpack(gdf[:].drop(target_name).to_dlpack())
    # if target is not None:
    #    tensors.append(target)
    return tensors, target

def load_tensors_from_parquet(path, target_name='delinquency_12'):
    tbl = pq.read_table(path).to_pandas()
    target = None
    if target_name in tbl:
        target = torch.from_numpy(tbl.pop(target_name).values.astype(np.float32))
    features = torch.from_numpy(tbl.values.astype(np.long))
    tensors = [features]
    if target is not None:
        tensors.append(target)
    return tuple(tensors)


class ParquetBatchDataset(object):
    def __init__(self, root_path, batch_size=1, num_files=None, target_name='delinquency_12',
                 use_cuDF=True, file_offset=0):
        self.batch_size=batch_size
        self.parquet_files = glob.glob(os.path.join(root_path, "*.parquet"))
        self.target_name = target_name

        num_files = self.parquet_files if num_files is None else num_files
        i = 0
        self.targets = None
        self.features = None
        for f in self.parquet_files:
            if i >= file_offset:
                if i >= num_files+file_offset: break
                print('loading: '+ f + ' into gpu memory')
                if use_cuDF:
                    feature, target = load_tensors_from_parquet_via_cudf(f)
                else: 
                    feature, target = load_tensors_from_parquet(f)
                if self.targets is None:
                    self.targets = target
                else:
                    self.targets = torch.cat((self.targets, target))
                if self.features is None:
                    self.features = feature
                else:
                    self.features = torch.cat((self.features, feature))
                i = i + 1
            else:
                i = i + 1

        self.num_samples = len(self.features)
        print('Total samples: ', self.num_samples)    
        if use_cuDF is False:
            self.features = self.features.cuda()
            self.targets = self.targets.cuda()
        
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        idx = item*self.batch_size
        #Need to handle odd sized batches if data isn't divisible by batchsize
        if idx+self.batch_size > self.num_samples:
            #if idx > self.num_samples:
                #raise StopIteration
            #else:
                return self.features[idx:], self.targets[idx:]
        return self.features[idx:idx+self.batch_size], self.targets[idx:idx+self.batch_size]
    
    def shuffle(self):
        idx = torch.randperm(self.num_samples)
        self.features = self.features[idx]
        self.targets = self.targets[idx]

        
class RandomLongBatchDataset(object):
    def __init__(self, num_samples=100000, batch_size=1, cpu_mem=True):
        self.batch_size=batch_size
        self.targets = None
        self.features = None

        self.targets = torch.LongTensor(num_samples).random_(0, 1).type(torch.FloatTensor)
        self.features = torch.LongTensor(num_samples, 45).random_(0, 2**22)
        self.num_samples = num_samples
        if cpu_mem is False:
            self.features = self.features.cuda()
            self.targets = self.targets.cuda()
        
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        idx = item*self.batch_size
        #Need to handle odd sized batches if data isn't divisible by batchsize
        if idx+self.batch_size > self.num_samples:
            #if idx > self.num_samples:
                #raise StopIteration
            #else:
                return self.features[idx:], self.targets[idx:]
        return self.features[idx:idx+self.batch_size], self.targets[idx:idx+self.batch_size]
    
    def shuffle(self):
        idx = torch.randperm(self.num_samples)
        self.features = self.features[idx]
        self.targets = self.targets[idx]        
        
class MultiRandomLongBatchDataset(object):
    def __init__(self, num_samples=100000, batch_size=1, cpu_mem=True):
        self.batch_size=batch_size
        self.targets = None
        self.features = None

        self.targets = torch.LongTensor(num_samples).random_(0, 1).type(torch.FloatTensor)
        self.split0 = num_samples//3 + batch_size-(num_samples//3)%batch_size
        self.split1 = 2*num_samples//3 + batch_size-(2*num_samples//3)%batch_size
        self.features0 = torch.LongTensor(self.split0, 45).random_(0, 2**22)
        self.features1 = torch.LongTensor(self.split1-self.split0, 45).random_(0, 2**22)
        self.features2 = torch.LongTensor(num_samples-self.split1, 45).random_(0, 2**22)
        self.num_samples = num_samples
        if cpu_mem is False:
            self.features0 = self.features0.cuda()
            self.features1 = self.features1.cuda()
            self.features2 = self.features2.cuda()
            self.targets = self.targets.cuda()
        
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        idx = item*self.batch_size
        #Need to handle odd sized batches if data isn't divisible by batchsize
        
        if idx < self.split0:
            return self.features0[idx:idx+self.batch_size], self.targets[idx:idx+self.batch_size]
        elif idx < self.split1:
            return self.features1[idx-self.split0:idx-self.split0+self.batch_size], self.targets[idx:idx+self.batch_size]
        else:
            if idx+self.batch_size > self.num_samples:
                return self.features2[idx-self.split1:], self.targets[idx:]
            else:
                return self.features2[idx-self.split1:idx-self.split1+self.batch_size], self.targets[idx:idx+self.batch_size]
            
    
    def shuffle(self):
        #idx = torch.randperm(self.num_samples)
        #self.features = self.features[idx]
        #self.targets = self.targets[idx]
        pass

class ConcatRandomLongBatchDataset(object):
    def __init__(self, num_samples=100000, batch_size=1, cpu_mem=True):
        self.batch_size=batch_size
        self.targets = None
        self.features = None

        self.targets = torch.LongTensor(num_samples).random_(0, 1).type(torch.FloatTensor)
        self.split0 = num_samples//3 + batch_size-(num_samples//3)%batch_size
        self.split1 = 2*num_samples//3 + batch_size-(2*num_samples//3)%batch_size
        self.features0 = torch.LongTensor(self.split0, 45).random_(0, 2**22)
        self.features1 = torch.LongTensor(self.split1-self.split0, 45).random_(0, 2**22)
        self.features2 = torch.LongTensor(num_samples-self.split1, 45).random_(0, 2**22)
        self.num_samples = num_samples
        if cpu_mem is False:
            self.features0 = self.features0.cuda()
            self.features1 = self.features1.cuda()
            self.features2 = self.features2.cuda()
            self.targets = self.targets.cuda()
        self.features = torch.cat([self.features0, self.features1, self.features2])
            
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        idx = item*self.batch_size
        #Need to handle odd sized batches if data isn't divisible by batchsize
        if idx+self.batch_size > self.num_samples:
            #if idx > self.num_samples:
                #raise StopIteration
            #else:
                return self.features[idx:], self.targets[idx:]
        return self.features[idx:idx+self.batch_size], self.targets[idx:idx+self.batch_size]
    
    def shuffle(self):
        idx = torch.randperm(self.num_samples)
        self.features = self.features[idx]
        self.targets = self.targets[idx]        
