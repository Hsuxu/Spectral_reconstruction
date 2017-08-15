import os 
import torch
import argparse
import time
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from data_util import *
from spec_model import *

parser = argparse.ArgumentParser(description='Spectral Super-resolution')
parser.add_argument('--batch_size', type=int, default=20,
                    help='input batch size for training (default: 8)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--seed', type=int, default=212,
                    metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--print_freq','-p',default=10,)
args = parser.parse_args()
args.cuda =torch.cuda.is_available()
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    
if not os.path.exists('weights/'):
    os.mkdir('weights/')

def train(epoch,model,train_loader):
    model.train()
    print('training...')
    for batch_idx,(data,target) in enumerate(train_loader):
        if args.cuda:
            data, target= data.cuda(),target.cuda()
        data=Variable(data)
        target=Variable(target)
        output=model(data)
        optimizer=optim.Adam(model.parameters(),lr=1e-3,betas=(0.9, 0.999))
        optimizer.zero_grad()
        criterion=nn.MSELoss().cuda()
        loss=criterion(output,target)
        loss.backward()
        optimizer.step()
        # psnr=compute_PSNR(loss.data[0])
        snr=compute_SNR(output,target)
        sam=compute_SAM(output,target)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch:{}/{} [{}/{} ({:.0f}%)]  MSEloss:{:.6f}  SNR:{:.4f}  SAM:{:.4f}'.format(
                epoch, args.epochs, batch_idx *len(data), len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader), loss.data[0],
                snr, sam
            ))

def test(epoch,model,test_loader):
    model.eval()
    print('testing...')
    test_loss = 0.0
    snr = 0.0
    sam = 0.0
    for i, (data, target) in enumerate(test_loader):
        if args.cuda:
            data,target=data.cuda(),target.cuda()
        data=Variable(data,volatile=True)
        target=Variable(target)
        output=model(data)
        criterion=nn.MSELoss().cuda()
        test_loss+=criterion(output,target).data[0]
        snr+=compute_SNR(output,target)
        sam+=compute_SAM(output,target)
    test_loss/=len(test_loader)
    psnr=compute_PSNR(test_loss)
    snr/=len(test_loader)
    sam/=len(test_loader)
    print('Test Set: Average MSEloss: {:.6f}  PSNR: {:.4f}  Average SNR: {:.4f}  Average SAM: {:.4f}'.format(test_loss, psnr, snr,sam))
    return snr,sam

def main():
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = DataLoader(SpecDataset(PATH, NAME, ISZ, DOWN_CHN, train=True), 
                                batch_size=args.batch_size, 
                                shuffle=True,
                                **kwargs)
    test_loader=DataLoader(SpecDataset(PATH, NAME, ISZ, DOWN_CHN, train=False), 
                                        batch_size=5, 
                                        shuffle=True,
                                        **kwargs)
    model= Resc_Net(DOWN_CHN,Target_CHN)
    # print model
    if args.cuda:
        model.cuda()

    best_snr=0.0
    for epoch in xrange(1,args.epochs):
        t1=time.time()
        train(epoch,model,train_loader)
        snr,sam=test(epoch,model,test_loader)
        if best_snr<snr:
            best_snr=snr
            save_checkpoint({
                'epoch':epoch,
                'arch':'Spectral_Reconstruction',
                'state_dict':model.state_dict(),
                'snr':snr,
                'sam':sam
            },filename='weights/ckp_spec_')
    torch.save(model.state_dict(),'weights/model.pth')
        # print loss
def save_checkpoint(state,filename='checkpoint'):
    torch.save(state,filename+'.pth.tar')

if __name__ == '__main__':
    main()
    

