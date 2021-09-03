import math
import torch
import torch.nn as nn

class GELU(nn.Module):
    def forward(self, tensor):
        return 0.5*tensor*(1+torch.tanh(math.sqrt(2 / math.pi) * (tensor+ 0.044715 * torch.pow(tensor, 3))))