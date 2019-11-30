import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np

class AdaptiveFeatureSelection(nn.Module):
    ''' AdaptiveFeatureSelection '''

    def __init__(self, down_num,down_ins,up_num,up_ins,cur_in,lang_in,hiddens,outs):
        super().__init__()
        self.afs_modules=[]
        for i in range(down_num):
            self.afs_modules.append(FeatureNormalize(down_ins[i],hiddens,outs,down_sample=True,scale_factor=2**(down_num-i)))
        self.afs_modules.append(FeatureNormalize(cur_in,hiddens,outs))
        for i in range(up_num):
            self.afs_modules.append(FeatureNormalize(up_ins[i],hiddens,outs,up_sample=True,scale_factor=2**(i+1)))
        self.afs_modules=nn.ModuleList(self.afs_modules)
        self.afs_weights=nn.Linear(lang_in,down_num+1+up_num)
    def forward(self, *input):
        lang=input[0]
        visuals=input[1]
        v_len=len(visuals)

        # print(self.afs_modules)
        for i in range(v_len):
            # print(i)
            # print(visuals[i].size())
            visuals[i]=self.afs_modules[i](visuals[i]).unsqueeze(-1)
            # print(visuals[i].size())
        v_size=visuals[0].size()
        visuals=torch.cat(visuals,-1).permute(0,4,1,2,3).contiguous().view(v_size[0],v_len,-1)
        # print(visuals.size())
        weights=self.afs_weights(lang)
        weights=F.softmax(weights,dim=-1).unsqueeze(1)
        # print(weights.size())
        outputs=torch.bmm(weights,visuals).view(v_size[:-1])
        # print(outputs.size())
        return outputs

class FeatureNormalize(nn.Module):
    ''' FeatureNormalize '''

    def __init__(self,ins,hiddens,outs,down_sample=False,up_sample=False,scale_factor=1.):
        super().__init__()
        self.normalize=None
        if down_sample:
            self.normalize=nn.AvgPool2d(scale_factor)
        elif up_sample:
            self.normalize = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        self.conv1=nn.Conv2d(ins, hiddens, 3, padding=1)
        self.norm1=nn.BatchNorm2d(hiddens)
        self.act1=nn.LeakyReLU(0.1, inplace=True)
        self.conv2=nn.Conv2d(hiddens, outs, 1)
        self.norm2=nn.BatchNorm2d(outs)
        self.act2=nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        if self.normalize is not None:
            x=self.normalize(x)
        x=self.conv1(x)
        x=self.norm1(x)
        x=self.act1(x)
        x=self.conv2(x)
        x=self.norm2(x)
        x=self.act2(x)
        return x
