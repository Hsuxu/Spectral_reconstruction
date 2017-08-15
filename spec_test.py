import os
import time
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_util import *
from spec_model import *

TEST_PATH='../data/'
TEST_NAME='Pavia_center.tif'
weights='weights/ckp_spec_.pth.tar'

# args.cuda =torch.cuda.is_available()

class SpecTest(DATA.Dataset):
    def __init__(self,PATH,NAME,ksize,down_chn):
        self.ksize = ksize
        self.data = tiff.imread(os.path.join(PATH, NAME)).astype(np.float32)
        self.data = samele_wise_normalization(self.data)
        self.data = spec_down_sample(self.data)
        self.shape = self.data.shape
        if self.shape[0]% self.ksize!=0:
            pad_width=self.ksize*(self.shape[0]//self.ksize+1)-self.shape[0]
            self.data=np.pad(self.data,((0,pad_width),(0,0),(0,0)),mode='symmetric')
        if self.shape[1]% self.ksize!=0:
            pad_width=self.ksize*(self.shape[1]//self.ksize+1)-self.shape[1]
            self.data=np.pad(self.data,((0,0),(0,pad_width),(0,0)),mode='symmetric')
        self.new_shape = self.data.shape
        self.Nh = self.new_shape[0] / self.ksize
        self.Mw = self.new_shape[1] / self.ksize
        idx = []
        idy = []
        for i in xrange(self.Nh):
            for j in xrange(self.Mw):
                idx.append(i)
                idy.append(j)
        self.loc = zip(idx, idy)
        
    def __getitem__(self, index):
        idx = self.loc[index][0]
        idy = self.loc[index][1]
        data = self.data[idx * self.ksize:(idx + 1) * self.ksize,
                         idy * self.ksize:(idy + 1) * self.ksize, :]
        data = np.transpose(data, [2, 0, 1])
        return data
    def __len__(self):
        return self.Nh*self.Mw

def test(model,test_loader):
    out_data=[]
    for i,(data) in enumerate(test_loader):
        # print(type(data))
        if torch.cuda.is_available():
            data=data.cuda()
        data=Variable(data,volatile=True)
        output=model(data)
        out_data.append(output.data.cpu().numpy())
    # print output
    return out_data

def cvt_data(out_list,dataset):
    
    if len(dataset)!=len(out_list):
        raise ValueError('The output is not match with dataset size')
    results = np.zeros(shape=[dataset.new_shape[0],dataset.new_shape[1],Target_CHN])
    ksize = dataset.ksize
    index = 0
    for i in xrange(dataset.Nh):
        for j in xrange(dataset.Mw):
            tmp= np.transpose( np.squeeze(out_list[index]),[1,2,0])
            results[i*ksize:(i+1)*ksize,j*ksize:(j+1)*ksize,:]=tmp
            index += 1
    shape = dataset.shape
    return inv_transform(results[:shape[0], :shape[1], :],)

def main():
    kwargs = {'num_workers': 2, 'pin_memory': True} if torch.cuda.is_available() else {}
    test_dataset = SpecTest(TEST_PATH, TEST_NAME, ISZ, DOWN_CHN)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             **kwargs
                             )
    model=Resc_Net(DOWN_CHN,Target_CHN)
    model.load_state_dict(torch.load(weights)['state_dict'])
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    cudnn.benchmark=True
    t1=time.time()
    results=test(model,test_loader)
    results=cvt_data(results,test_dataset)
    print results.shape
    tiff.imsave(TEST_PATH+'Pavia_rec.tif',results)
    print('Elapsed time:{:.2f}s'.format(time.time()-t1))
if __name__ == '__main__':
    main()


# for data in test_loader:
#     print(data.shape)
