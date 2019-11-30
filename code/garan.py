import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np
class CollectDiffuseAttention(nn.Module):
    ''' CollectDiffuseAttention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout_c = nn.Dropout(attn_dropout)
        self.dropout_d = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)


    def forward(self, q, kc,kd, v, mask=None):
        '''
        q: n*b,1,d_o
        kc: n*b,h*w,d_o
        kd: n*b,h*w,d_o
        v: n*b,h*w,d_o
        '''

        attn_col = torch.bmm(q, kc.transpose(1, 2)) #n*b,1,h*w
        attn_col_logit = attn_col / self.temperature
        attn_col = self.softmax(attn_col_logit)
        attn_col = self.dropout_c(attn_col)
        attn = torch.bmm(attn_col, v) #n*b,1,d_o

        attn_dif = torch.bmm(kd,q.transpose(1, 2)) #n*b,h*w,1
        attn_dif_logit = attn_dif / self.temperature
        attn_dif = F.sigmoid(attn_dif_logit)
        attn_dif= self.dropout_d(attn_dif)
        output=torch.bmm(attn_dif,attn)
        return output, attn_col_logit.squeeze(1)
class GaranAttention(nn.Module):
    ''' GaranAttention module '''

    def __init__(self,d_q, d_v,n_head=2, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_q = d_q
        self.d_v = d_v
        self.d_k=d_v
        self.d_o=d_v
        d_o=d_v

        self.w_qs = nn.Linear(d_q, d_o,1)
        self.w_kc = nn.Conv2d(d_v, d_o,1)
        self.w_kd = nn.Conv2d(d_v, d_o,1)
        self.w_vs = nn.Conv2d(d_v, d_o,1)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_q + d_o)))
        nn.init.normal_(self.w_kc.weight, mean=0, std=np.sqrt(2.0 / (d_v + d_o//n_head)))
        nn.init.normal_(self.w_kd.weight, mean=0, std=np.sqrt(2.0 / (d_v + d_o//n_head)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_v + d_o//n_head)))

        self.attention = CollectDiffuseAttention(temperature=np.power(d_o//n_head, 0.5))
        self.layer_norm = nn.BatchNorm2d(d_o)
        self.layer_acti= nn.LeakyReLU(0.1,inplace=True)

        # nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, v, mask=None):

        d_k, d_v, n_head,d_o = self.d_k, self.d_v, self.n_head,self.d_o

        sz_b, c_q = q.size()
        sz_b,c_v, h_v,w_v = v.size()
        # print(v.size())
        residual = v

        q = self.w_qs(q)
        kc=self.w_kc(v).view(sz_b,n_head,d_o//n_head,h_v*w_v)
        kd=self.w_kd(v).view(sz_b,n_head,d_o//n_head,h_v*w_v)
        v=self.w_vs(v).view(sz_b,n_head,d_o//n_head,h_v*w_v)
        q=q.view(sz_b,n_head,1,d_o//n_head)
        # v=v.view(sz_b,h_v*w_v,n_head,c_v//n_head)

        q = q.view(-1, 1, d_o//n_head) # (n*b) x lq x dk
        kc = kc.permute(0,1,3,2).contiguous().view(-1, h_v*w_v, d_o//n_head) # (n*b) x lk x dk
        kd=kd.permute(0,1,3,2).contiguous().view(-1, h_v*w_v, d_o//n_head) # (n*b) x lk x dk
        v = v.permute(0,1,3,2).contiguous().view(-1, h_v*w_v, d_o//n_head) # (n*b) x lv x dv

        output, attn = self.attention(q, kc,kd, v)
        #n * b, h * w, d_o
        output = output.view(sz_b,n_head, h_v,w_v, d_o//n_head)
        output = output.permute(0,1,4,3,2).contiguous().view(sz_b,-1, h_v,w_v) # b x lq x (n*dv)
        attn=attn.view(sz_b,n_head, h_v,w_v)
        attn=attn.mean(1)
        #residual connect
        output= output+residual
        output=self.layer_norm(output)
        output=self.layer_acti(output)

        # output = self.dropout(self.fc(output))

        return output, attn