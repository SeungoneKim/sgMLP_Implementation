import torch 
import torch.nn as nn 
import torch.nn.functional as F
from CLS_model.embedding import TransformerEmbedding
from CLS_model.layer import gMLPBLOCK_CLS
import horovod.torch as hvd


class gMLP(nn.Module):
    def __init__(self,d_model,d_ffn,seq_len,num_layers):
        super(gMLP,self).__init__()
        self.model = nn.Sequential(*[gMLPBLOCK_CLS(d_model,d_ffn,seq_len) for _ in range(num_layers)])
        
    def forward(self,x):
        x = self.model(x)
        return x

class NaturalLanguageUnderstandingHead(nn.Module):
    def __init__(self, vocab_size, model_dim):
        super(NaturalLanguageUnderstandingHead,self).__init__()
        self.linear_layer = nn.Linear(model_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, encoder_output):
        # mask_position = [bs, tgt_size(15% of sent)]
        mlm_prediction = self.softmax(self.linear_layer(encoder_output)) # [bs,sl,vocab_size]
        
        return mlm_prediction

class gMLP_LanguageModel(gMLP):
    def __init__(self,vocab_size, d_model, d_ffn, seq_len, num_layers,output_logits=False):
        super().__init__(d_model,d_ffn,seq_len,num_layers)
        self.embed = TransformerEmbedding(vocab_size,d_model,seq_len,0.1)
        self.output_logits = output_logits
        self.to_logits = NaturalLanguageUnderstandingHead(vocab_size,d_model)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self,x):
        embedding = self.embed(x)
        embedding = embedding.cuda()
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
def build_model(num_tokens, d_model, d_ffn, seq_len, num_layers):
    
    model = gMLP_LanguageModel(num_tokens,d_model,d_ffn,
                            seq_len,num_layers,True)
    if torch.cuda.is_available():
        torch.cuda.set_device(hvd.local_rank())
        model.cuda()
    return model
