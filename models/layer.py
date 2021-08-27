import torch 
import torch.nn as nn 
import torch.nn.functional as F
from models.embedding import TransformerEmbedding
from util.gelu import GELU

class gMLPBLOCK(nn.Module):
    def __init__(self,d_model,d_ffn,seq_len):
        super(gMLPBLOCK,self).__init__()

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.channel_proj_i = nn.Linear(d_model,d_ffn*2)
        self.channel_proj_ii = nn.Linear(d_ffn,d_model)
        self.sgu = SpatialGatingUnit(d_ffn,seq_len)

    def forward(self,x):
        residual = x
        x = self.layer_norm(x)
        x = F.gelu(self.channel_proj_i(x))
        x = self.sgu(x)
        x = self.channel_proj_ii(x)

        return residual + x

class SpatialGatingUnit(nn.Module):
    def __init__(self,d_ffn,seq_len):
        super(SpatialGatingUnit,self).__init__()
        self.layer_norm = nn.LayerNorm(d_ffn)
        self.spatial_token = SpatialTokenGen(d_ffn,seq_len)
        self.spatial_proj_i = nn.Conv1d(seq_len,seq_len,1)
        self.spatial_proj_ii = nn.Conv1d(seq_len,seq_len,1)
        nn.init.constant_(self.spatial_proj_i.bias, 1.0)
        nn.init.constant_(self.spatial_proj_ii.bias, 1.0)

    def forward(self,x):
        residual = x #학습이 느려질 여지가 될 수도 residual / 파라미터 증가 
        u,v = x.chunk(2,dim=-1)
        u = self.spatial_token(u)
        v = self.layer_norm(v)
        if u>0.5:
            v = self.spatial_proj_i(v)
        else:
            v = self.spatial_proj_ii(v)
        return v
    
class SpatialTokenGen(nn.Module):
    def __init__(self,d_ffn,seq_len):
        super(SpatialTokenGen,self).__init__()
        self.layer_norm = nn.LayerNorm(d_ffn)
        self.squeeze_layer_i = nn.Linear(d_ffn,1)
        self.squeeze_layer_ii = nn.Conv1d(seq_len,1,1)

    def forward(self,x):
        x = self.layer_norm(x)
        x = self.squeeze_layer_i(x)
        x = self.squeeze_layer_ii(x)
        tok = torch.mean(x)
        tok = torch.sigmoid(tok)
        return tok