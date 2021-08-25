import torch 
import torch.nn as nn 
import torch.nn.functional as F

class gMLPBLOCK(nn.Module):
    def __init__(self,d_model,d_ffn,seq_len):
        super(gMLPBLOCK,self).__init__()

        self.layer_norm = nn.LayerNorm(d_model)
        self.channel_proj_i = nn.Linear(d_model,d_ffn)
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
        self.spatial_cls = nn.Linear(d_ffn,1)
        self.spatial_proj_i = nn.Conv1d(seq_len,seq_len,1)
        self.spatial_proj_ii = nn.Conv1d(seq_len,seq_len,1)
        nn.init.constant_(self.spatial_proj_i.bias, 1.0)
        nn.init.constant_(self.spatial_proj_ii.bias, 1.0)

    def forward(self,x):
        residual = x #학습이 느려질 여지가 될 수도 residual / 파라미터 증가 
        cls  = torch.stack([x[n][0] for n in range(x.shape[0])])
        cls  = self.spatial_cls(cls).squeeze()
        cls = torch.sigmoid(cls)
        x = self.layer_norm(x)
        if cls>0.5:
            x = self.spatial_proj_i(x)
        else:
            x = self.spatial_proj_ii(x)
        return x+residual
    
class gMLP(nn.Module):
    def __init__(self,d_model,d_ffn,seq_len,num_layers):
        super(gMLP,self).__init__()
        self.model = nn.Sequential(*[gMLPBLOCK(d_model,d_ffn,seq_len) for _ in range(num_layers)])
        
    def forward(self,x):
        x = self.model(x)
        return x

class gMLP_LanguageModel(gMLP):
    def __init__(self,vocab_size, d_model, d_ffn, seq_len, num_layers,device,output_logits=False):
        super().__init__(d_model,d_ffn,seq_len,num_layers)
        self.device = device
        self.embed = nn.Embedding(vocab_size,d_model)
        self.output_logits = output_logits
        self.to_logits = nn.Sequential(nn.LayerNorm(d_model),nn.Linear(d_model,vocab_size))
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self,x):
        embedding = self.embed(x)
        embedding = embedding.to(self.device)
        output = self.model(embedding)
        if self.output_logits:
            output = self.softmax(self.to_logits(output))


        return output
    
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
"""
tmp_model = build_model(28996,256,1024,256,4)
params = list(tmp_model.parameters())
print("The number of parameters:",sum([p.numel() for p in tmp_model.parameters() if p.requires_grad]), "elements")
The number of parameters: 18580300 elements
"""
def build_model(num_tokens, d_model, d_ffn, seq_len, num_layers,device):
    
    model = gMLP_LanguageModel(num_tokens,d_model,d_ffn,
                            seq_len,num_layers,device,True).to(device)
    
    if torch.cuda.device_count()>1:
        print("Using ",torch.cuda.device_count(),"GPUs in total!")
        model = torch.nn.DataParallel(model,device_ids=[0,1,2,3],output_device=1)
    
    return model.cuda() if torch.cuda.is_available() else model
