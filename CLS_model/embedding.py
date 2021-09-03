import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, model_dim):
        super(TokenEmbedding, self).__init__(vocab_size, model_dim, padding_idx=0)

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len, device):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, model_dim, device=device).float()
        pe.requires_grad = False
        
        pos = torch.arange(0,max_len,device=device).float().unsqueeze(dim=1)
        divterm = (torch.arange(0,model_dim,step=2,device=device).float() * -(math.log(10000.0) / model_dim)).exp()
        
        # pe = (1, sequence_length, hidden_size)
        pe[:, 0::2] = torch.sin(pos * divterm)
        pe[:, 1::2] = torch.cos(pos * divterm)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)
        
    def forward(self, tensor):
        
        # (1, sequence_length, hidden_size)
        return self.pe[:, :tensor.size(1)]

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, model_dim, max_len, drop_prob, device):
        super(TransformerEmbedding,self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, model_dim)
        self.pos_emb = PositionalEncoding(model_dim, max_len, device)
        #self.tok_drop_out = nn.Dropout(drop_prob)
        #self.seg_drop_out = nn.Dropout(drop_prob)
        self.dropout = nn.Dropout(drop_prob)
        self.device = device
    
    def forward(self, tensor):
        tok_emb = self.tok_emb(tensor)
        pos_emb = self.pos_emb(tensor)

        tok_emb = tok_emb.to(self.device)
        pos_emb = pos_emb.cuda(tok_emb.device)
        
        return self.dropout(tok_emb + pos_emb)
