import torch
import torch.nn as nn
import torch.nn.functional as F
from data_util import *

class Fire(nn.Module):
    def __init__(self,input_chn,squeeze_chn,ex_chn):
        super(Fire,self).__init__()
        self.Conv1=nn.Conv2d(input_chn,squeeze_chn,kernel_size=3,stride=1,padding=1)
        # self.bn1=nn.BatchNorm2d(squeeze_chn)
        self.ex1=nn.Conv2d(squeeze_chn,ex_chn,kernel_size=1,stride=1,padding=0)
        self.ex3=nn.Conv2d(squeeze_chn,ex_chn,kernel_size=3,stride=1,padding=1)
        self.Conv2=nn.Conv2d(ex_chn*2,squeeze_chn,kernel_size=1,stride=1,padding=0)
    def forward(self,x):
        x=F.relu(self.Conv1(x))
        x1=self.ex1(x)
        x2=self.ex3(x)
        x=torch.cat([x1,x2],dim=1)
        out=self.Conv2(x)
        return out

class Down_Transition(nn.Module):
    def __init__(self, input_chn,out_chn):
        super(Down_Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(input_chn)
        self.down=nn.AvgPool2d(kernel_size=2)
        self.act = nn.ReLU(inplace=True)
        self.Conv1=nn.Conv2d(input_chn,out_chn,kernel_size=1,stride=1,padding=0)
    def forward(self,x):
        x=self.bn1(x)
        x=self.down(x)
        return self.Conv1(self.act(x))

class Up_Transition(nn.Module):
    def __init__(self,input_chn,out_chn):
        super(Up_Transition,self).__init__()
        self.bn1 = nn.BatchNorm2d(input_chn)
        self.act1 = nn.ReLU(inplace=True)
        self.DeConv1=nn.ConvTranspose2d(input_chn,input_chn,kernel_size=3,stride=2,padding=1,output_padding=1)
        # self.bn2 = nn.BatchNorm2d(input_chn)
        self.act2 = nn.ReLU(inplace=True)
        self.Conv2=nn.Conv2d(input_chn,out_chn,kernel_size=1,stride=1,padding=0)
    def forward(self,x):
        x=self.act1(self.bn1(x))
        x=self.act2(self.DeConv1(x))
        return self.Conv2(x)

class Resc_Net(nn.Module):
    def __init__(self,data_chn,target_chn):
        super(Resc_Net,self).__init__()
        filters=[32,64,128,256,512]
        self.Conv0 = nn.Conv2d(
            data_chn, filters[0], kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(filters[0])
        
        self.block1 = self._down_block(
            filters[0], filters[2], filters[1], out_chn=filters[1])
        
        self.block2 = self._down_block(
            filters[1], filters[3], filters[2], filters[2])
        
        self.Trans1 = nn.Conv2d(
            filters[2], filters[1], kernel_size=1, stride=1, padding=0)
        
        self.up1 = self._up_block(
            filters[1], filters[3], filters[2], out_chn=filters[2])
        self.Trans2 = nn.Conv2d(filters[1] + filters[2], filters[2],kernel_size=1)
        
        self.up2 = self._up_block(filters[2],filters[4],filters[3],filters[3])
        self.Conv2=nn.Conv2d(filters[3],filters[4],kernel_size=3,stride=1,padding=1)
        self.Trans3=nn.Conv2d(filters[4],target_chn,kernel_size=1)

    def forward(self, x):
        x0 = F.relu(self.bn1(self.Conv0(x)))  # 64
        xd1 = self.block1(x0)  # 32
        xd2 = self.block2(xd1)  # 16
        xd2 = self.Trans1(xd2)  # 16
        xu1 = self.up1(xd2)  # 32
        xu1 = self.Trans2(torch.cat([xd1, xu1], dim=1))  # 32
        xu2 = self.up2(xu1)
        out = F.relu(self.Trans3(self.Conv2(xu2)))
        return out

    def _down_block(self, input_chn, inner_chn, ex_chn, out_chn):
        layers = []
        layers.append(Fire(input_chn, inner_chn, ex_chn))
        layers.append(Down_Transition(inner_chn, out_chn))
        return nn.Sequential(*layers)

    def _up_block(self, input_chn, inner_chn, ex_chn, out_chn):
        layers = []
        layers.append(Fire(input_chn, inner_chn, ex_chn))
        layers.append(Up_Transition(inner_chn, out_chn))
        return nn.Sequential(*layers)
