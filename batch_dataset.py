import torch

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
        if idx < self.num_samples-1 and idx > self.num_samples-1 - self.batch_size:
                return self.features[idx:], self.targets[idx:]
        return self.features[idx:idx+self.batch_size], self.targets[idx:idx+self.batch_size]
    
    def shuffle(self):
        print('shuffling batch')
        idx = torch.randperm(self.num_samples, dtype=torch.int64, device='cuda')
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
        idx = torch.randperm(self.num_samples, dtype=torch.int64, device='cuda')
        idx0 = torch.randperm(self.split0, dtype=torch.int64, device='cuda')
        idx1 =  torch.randperm(self.split1-self.split0, dtype=torch.int64, device='cuda')
        idx2 =  torch.randperm(self.num_samples-self.split1, dtype=torch.int64, device='cuda')
        self.features0 = self.features0[idx0]
        self.features1 = self.features1[idx1]
        self.features2 = self.features2[idx2]
        self.targets = self.targets[idx]
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
        self.features = torch.cat(self.features0, self.features1, self.features2)
            
    
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
