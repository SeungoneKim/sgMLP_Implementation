import torch 
import torch.nn as nn 
import torch.nn.functional as F
from CLS_model.embedding import TransformerEmbedding
from util.gelu import GELU

class gMLPBLOCK_CLS(nn.Module):
    def __init__(self,d_model,d_ffn,seq_len):
        super(gMLPBLOCK_CLS,self).__init__()

        self.layer_norm = nn.LayerNorm(d_model)
        self.channel_proj_i = nn.Linear(d_model,d_ffn)
        self.channel_proj_ii = nn.Linear(d_ffn,d_model)
        self.sgu = SpatialGatingUnit_CLS(d_ffn,seq_len)

    def forward(self,x):
        residual = x
        x = self.layer_norm(x)
        x = F.gelu(self.channel_proj_i(x))
        x = self.sgu(x)
        x = self.channel_proj_ii(x)
        return residual + x

class SpatialGatingUnit_CLS(nn.Module):
    def __init__(self,d_ffn,seq_len):
        super(SpatialGatingUnit_CLS,self).__init__()
        self.layer_norm = nn.LayerNorm(d_ffn)
        self.spatial_cls = nn.Linear(d_ffn,1)
        self.spatial_proj_i = nn.Conv1d(seq_len,seq_len,1)
        self.spatial_proj_ii = nn.Conv1d(seq_len,seq_len,1)
        nn.init.constant_(self.spatial_proj_i.bias, 1.0)
        nn.init.constant_(self.spatial_proj_ii.bias, 1.0)

    def forward(self,x):
        residual = x #학습이 느려질 여지가 될 수도 residual / 파라미터 증가 
        cls  = torch.stack([x[n][0] for n in range(x.shape[0])])
        cls  = torch.mean(self.spatial_cls(cls).squeeze())
        cls = torch.sigmoid(cls)
        x = self.layer_norm(x)
        print(cls)
        if cls>0.5:
            x = self.spatial_proj_i(x)
        else:
            x = self.spatial_proj_ii(x)
        return x+residual
