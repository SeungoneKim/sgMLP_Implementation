import torch 
import torch.nn as nn 
import torch.nn.functional as F
from models.embedding import TransformerEmbedding
from models.layer import gMLPBLOCK, SpatialGatingUnit, SpatialTokenGen
import numpy as np

class gMLP(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len, num_layers):
        super(gMLP,self).__init__()
        self.model = nn.Sequential(*[gMLPBLOCK(d_model, d_ffn, seq_len) for _ in range(num_layers)])
        
    def forward(self, x):
        x = self.model(x)
        return x

class NaturalLanguageUnderstandingHead(nn.Module):
    def __init__(self, vocab_size, model_dim, device):
        super(NaturalLanguageUnderstandingHead,self).__init__()
        self.linear_layer = nn.Linear(model_dim, vocab_size).to(device)
        self.softmax = nn.LogSoftmax( dim=-1 )
    
    def forward(self, encoder_output):
        # mask_position = [bs, tgt_size(15% of sent)]
        mlm_prediction = self.softmax(self.linear_layer(encoder_output)) # [bs,sl,vocab_size]
        
        return mlm_prediction

# NLLLoss => requires LogSoftmax at last
class gMLP_LanguageModel(gMLP):
    def __init__(self,vocab_size, d_model, d_ffn, seq_len, num_layers, device, output_logits = False):
        super().__init__(d_model, d_ffn, seq_len, num_layers)
        self.device = device
        self.embed = TransformerEmbedding(vocab_size, d_model, seq_len, 0.1, device)
        self.output_logits = output_logits
        self.to_logits = NaturalLanguageUnderstandingHead(vocab_size, d_model, device)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self,x, token_type_ids):
        embedding = self.embed(x, token_type_ids)
        embedding = embedding.to(self.device)
        output = self.model(embedding)
        if self.output_logits:
            output = self.to_logits(output)

        return output

# CrossEntropyLoss
class OneSentenceClassificationHead(nn.Module):
    """
    for cola, sst2
    output : [bs, 2] => p(True), p(False)
    """
    def __init__(self, input_dim, inner_dim, pooler_dropout):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = nn.GELU()
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, 2)

    def forward(self, feature):
        # feature : model(body) output
        x = feature[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

# CrossEntropyLoss
class TwoSentenceClassificationHead(nn.Module):
    """
    for rte, mrpc, qqp, mnli
    output : [bs, 2]
    """
    def __init__(self,input_dim, inner_dim, pooler_dropout):
        """

        input_dim: hidden_dim * 2 (!!! be careful !!!  two [cls] tokens will be concatenated)
        inner_dim: hidden_dim

        """
        super().__init__()
        self.ffn = nn.Sequential(nn.Tanh(),nn.Linear(input_dim, inner_dim,),
                                 nn.ReLU(),nn.Linear(inner_dim, 1))

        '''
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = nn.GELU()
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, 2)
        '''
    def forward(self, feature, second_cls_idx):

        indices = torch.tensor([0, second_cls_idx])  # indices for selecting 1st and 2nd [cls]
        cls1 = feature[:,0,:] # first [cls]

        cls2 = feature[second_cls_idx[0],second_cls_idx[1]]  # access two [cls] tokens embedding
        x = torch.cat([cls1, cls2], dim=-1)
        # x : [bs , 2 ,hidden_dim]

        x = x.view(x.shape[0], -1)
        # concat two [cls] token embedding
        # x: [bs, hidden_dim * 2]

        x = self.ffn(x)
        return x



############################################################
#  STS-B what i have to do
class TwoSentenceRegressionHead(nn.Module):
    """
    for sts-b
    output : [1] (cosine_similarity * 5)
    """
    def __init__(self, input_dim, inner_dim, pooler_dropout, cos_eps):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.preserve = nn.Linear(input_dim,input_dim)
        torch.nn.init.xavier_uniform_(self.preserve.weight)
        self.activation_fn = nn.ReLU() ## originally Tanh
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, 2)
        self.cos = nn.CosineSimilarity(dim=1, eps=cos_eps)

    def forward(self, feature, second_cls_idx): ## feature is lm output
        feature = self.preserve(feature)
        feature = self.activation_fn(feature)
        feature = self.dropout(feature)
        feature = self.preserve(feature)
        feature = self.activation_fn(feature)
        feature = self.dropout(feature)
        feature = self.preserve(feature)
        feature = self.activation_fn(feature)
        feature = self.dropout(feature)
        feature = self.preserve(feature)
        feature = self.activation_fn(feature)
        feature = self.dropout(feature)
        feature = self.preserve(feature)
        feature = self.activation_fn(feature)
        feature = self.dropout(feature)

        cls1 = feature[:, 0, :]  # first [cls]
        cls2 = feature[second_cls_idx[0], second_cls_idx[1]]  # second [cls] shape(64,512)
        # get cosine similarity between two [cls] tokens

        similarity = self.cos(cls1, cls2) * 5 #shpae(64)

        # return with multiplying 5
        return similarity
###############################################################





class gMLP_ClassificationModel(gMLP_LanguageModel):
    def __init__(self, vocab_size, d_model, d_ffn, seq_len, num_layers, device, output_logits=False, task_type="one"):
        super().__init__(vocab_size, d_model, d_ffn, seq_len, num_layers, device, False)
        self.device = device
        self.OneSentence = OneSentenceClassificationHead(d_model, int(d_model/2), 0.15)
        self.TwoSentence = TwoSentenceClassificationHead(d_model*2, d_model, 0.15)
        # "one" => cola, sst2
        # "two" => stsb, rte, mrpc, qqp, mnli
        self.task_type = task_type

    def forward(self, x, token_type_ids):
        embedding = self.embed(x, token_type_ids)
        embedding = embedding.to(self.device)
        output = self.model(embedding)
        if self.task_type == "one":
            output = self.OneSentence(output)
        if self.task_type == "two":
            output = self.TwoSentence(output)

        return output

class gMLP_RegressionModel(gMLP_LanguageModel):
    def __init__(self, vocab_size, d_model, d_ffn, seq_len, num_layers, device, output_logits=False, task_type="one"):
        super().__init__(vocab_size, d_model, d_ffn, seq_len, num_layers, device, False)
        self.device = device
        # stsb
        self.Regression = TwoSentenceRegressionHead(d_model, int(d_model/2), 0.15, 1e-8)
        

    def forward(self,x, token_type_ids):
        embedding = self.embed(x, token_type_ids)
        embedding = embedding.to(self.device)
        indices = (x == 101).nonzero()  # 101 is cls
        indices = indices.transpose(0, 1)
        indices = indices[:, indices[1] > 0]

        output = self.model(embedding)

        # cls 위치를 모델 쪽에서 seq index를 넘겨줘서#

        output = self.Regression(output, indices)

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

def build_classification_model(num_tokens, d_model, d_ffn, seq_len, num_layers,device, task_type):
    model = gMLP_ClassificationModel(num_tokens,d_model,d_ffn,
                            seq_len,num_layers,device,False,task_type).to(device)

    if torch.cuda.device_count()>1:
        print("Using ",torch.cuda.device_count(),"GPUs in total!")
        model = torch.nn.DataParallel(model,device_ids=[0,1,2,3],output_device=1)
    
    return model.cuda() if torch.cuda.is_available() else model

def build_regression_model(num_tokens, d_model, d_ffn, seq_len, num_layers, device):
    model = gMLP_RegressionModel(num_tokens, d_model, d_ffn, seq_len, num_layers, device, False).to(device)
    
    if torch.cuda.device_count()>1:
        print("Using ", torch.cuda.device_count(), "GPUs in total!")
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3], output_device=1)
    
    return model.cuda() if torch.cuda.is_available() else model