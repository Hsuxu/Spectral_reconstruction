import os
import torch
import cv2
import math
import torchvision.datasets
import numpy as np
import tifffile as tiff
import torch.utils.data as DATA
from torch.utils.data import DataLoader
PATH='../data/'
NAME='Pavia_center.tif'
Target_CHN=102
DOWN_CHN=50
ISZ=64
MAX=8000.0
MIN=0.0
N_Int=0

def samele_wise_normalization(data):
    """
    normalize each sample to 0-1
    Input:
        sample
    Output:
        Normalized sample
    x'=1.0*(x-np.min(x))/(np.max(x)-np.min(x))
    """
    data.astype(np.float32)
    if np.max(data) == np.min(data):
        return np.ones_like(data, dtype=np.float32) * 1e-6
    else:
        return 1.0 * (data - np.min(data)) / (np.max(data) - np.min(data))

def inv_transform(data,MA=8000.0,MI=0.0):
    """
    reverse transform the data in 0-1 to original scale
    """
    return (data*(MA-MI)+MI).astype(np.uint16)

def spec_down_sample(data,target_chn=50):
    """
    down sample the spectral axes
    """
    data=np.asarray(data,dtype=np.float32)
    data=np.transpose(data,[2,0,1])
    c,h,w=data.shape
    tmp=np.zeros((target_chn,h,w))
    for i in xrange(w):
        tmp[:,:,i]=cv2.resize(data[:,:,i], (h,target_chn),interpolation=cv2.INTER_CUBIC)
    return np.transpose(tmp,[1,2,0]).astype(np.float32)

def random_rot_flip(data,label):
    if np.random.random()<0.5:
        k=np.random.randint(1,4)
        data=np.rot90(data,k=k)
        label=np.rot90(label,k=k)
    if np.random.random()<0.5:
        data = np.fliplr(data)
        label = np.fliplr(label)
    if np.random.random()<0.5:
        data = np.flip(data, axis=1)
        label = np.flip(label, axis=1)
    return data.copy(),label.copy()

class SpecDataset(DATA.Dataset):
    def __init__(self, PATH, NAME, ksize,down_chn, train=True):
        self.path = PATH
        self.name = NAME
        self.down_chn = down_chn
        self.train = train
        self.ksize = ksize
        self.y_train = tiff.imread(os.path.join(self.path, self.name)).astype(np.float32)

        self.y_train=samele_wise_normalization(self.y_train)
        self.X_train = spec_down_sample(self.y_train,self.down_chn)
        if self.train:
            self.num = 2500
            idx = np.random.randint(
                0,
                int(self.y_train.shape[0] * 0.8) - self.ksize,
                size=self.num)
            idy = np.random.randint(
                0, self.y_train.shape[1] - self.ksize, size=self.num)
            self.location=zip(idx,idy)
        else:
            self.num = 200
            idx = np.random.randint(
                int(self.y_train.shape[0] * 0.8),
                self.y_train.shape[0] - self.ksize,
                size=self.num)
            idy = np.random.randint(
                0, self.y_train.shape[1] - self.ksize, size=self.num)
            self.location=zip(idx,idy)
    def __getitem__(self, index):
        idx=self.location[index][0]
        idy=self.location[index][1]
        data=self.X_train[idx:idx+self.ksize,idy:idy+self.ksize,:]
        label=self.y_train[idx:idx+self.ksize,idy:idy+self.ksize,:]
        if self.train:
            data,label=random_rot_flip(data,label)
        data=np.transpose(data,[2,0,1])
        label=np.transpose(label,[2,0,1])
        # print(data.shape,label.shape,data.strides)
        return data, label
    def __len__(self):
        return self.num
        
def compute_PSNR(MSEloss):
    Max=2.0**N_Int
    log_max=torch.mul(torch.log(torch.FloatTensor([Max])),2.0)
    log_mse=torch.log(torch.FloatTensor([MSEloss]))
    return (log_max-log_mse).tolist()[0]*10.0

def compute_SNR(output,target):
    """
    compute the SNR between the target and output data
    """
    log_tar=torch.log(torch.sum(torch.pow(target,2)))
    log_dif=torch.log(torch.sum(torch.pow((target-output),2)))
    # print(log_tar,log_dif)
    return (log_tar-log_dif).data[0]*10.0

def compute_SAM(output,target):
    if torch.norm(target).data[0]==0.0 or torch.norm(output).data[0]==0.0:
        s=0.0
    else:
        s=(torch.acos(output.dot(target)/torch.norm(output)/torch.norm(target))/math.pi)*180.0
    return s.data[0]
    

# dataset=SpecDataset(PATH,NAME,ISZ,DOWN_CHN,train=False)
# dataloader=DataLoader(dataset ,batch_size=1,shuffle=False)
# print type(dataset.X_train[0,0,0])
# print type(dataset.y_train[0,0,0])

# for i,(data,target) in enumerate(dataloader):
#     print i
#     # print dataset.location[i]
#     # print(data)
#     print(type(target))
# # dataloader=SpecDataset(PATH,NAME,ISZ,DOWN_CHN)
# loc=dataloader.location
# print(np.where(loc<0))
# print(dataloader.location[0][0])
# print(dataloader.location[0][1])
# print(dataloader.y_train.shape)
