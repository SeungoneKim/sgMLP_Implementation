import torch 
import torch.nn as nn 
import torch.nn.functional as F
from models.embedding import TransformerEmbedding
from models.layer import gMLPBLOCK, SpatialGatingUnit, SpatialTokenGen

class gMLP(nn.Module):
    def __init__(self,d_model,d_ffn,seq_len,num_layers):
        super(gMLP,self).__init__()
        self.model = nn.Sequential(*[gMLPBLOCK(d_model,d_ffn,seq_len) for _ in range(num_layers)])
        
    def forward(self,x):
        x = self.model(x)
        return x

class NaturalLanguageUnderstandingHead(nn.Module):
    def __init__(self, vocab_size, model_dim, device):
        super(NaturalLanguageUnderstandingHead,self).__init__()
        self.linear_layer = nn.Linear(model_dim, vocab_size).to(device)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, encoder_output):
        # mask_position = [bs, tgt_size(15% of sent)]
        mlm_prediction = self.softmax(self.linear_layer(encoder_output)) # [bs,sl,vocab_size]
        
        return mlm_prediction

class gMLP_LanguageModel(gMLP):
    def __init__(self,vocab_size, d_model, d_ffn, seq_len, num_layers,device,output_logits=False):
        super().__init__(d_model,d_ffn,seq_len,num_layers)
        self.device = device
        self.embed = TransformerEmbedding(vocab_size,d_model,seq_len,0.1,device)
        self.output_logits = output_logits
        self.to_logits = NaturalLanguageUnderstandingHead(vocab_size,d_model,device)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self,x, token_type_ids):
        embedding = self.embed(x, token_type_ids)
        embedding = embedding.to(self.device)
        output = self.model(embedding)
        if self.output_logits:
            output = self.softmax(self.to_logits(output))

        return output
    
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