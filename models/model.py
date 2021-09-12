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

# NLLLoss => requires LogSoftmax at last
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
            output = self.to_logits(output)

        return output

# BCELoss
class OneSentenceClassificationHead(nn.Module):
    """
    for cola, sst2
    output : [bs, 1] => p(True), p(False)
    """
    def __init__(self, input_dim, inner_dim, pooler_dropout, device):
        super().__init__()
        self.pooler = nn.Linear(input_dim,input_dim).to(device)
        self.layer1 = nn.Linear(input_dim, inner_dim).to(device)
        self.layer2 = nn.Linear(inner_dim, inner_dim).to(device)
        self.projection = nn.Linear(inner_dim,1).to(device)
        self.layernorm1 = nn.BatchNorm1d(inner_dim).to(device)
        self.layernorm2 = nn.BatchNorm1d(inner_dim).to(device)
        self.dropout = nn.Dropout(p=pooler_dropout).to(device)
        self.relu = nn.ReLU().to(device)


    def forward(self, feature):
        # 헤드 성능이 의심될때 주석해제하고 parameter, gradient값 볼 수 있습니다.
        # for params in self.projection.parameters():
        #     print(params if len(params) < 10 else params[:10])
        #     print("#"*10)
        #     if params.grad != None:
        #         print(params.grad if len(params.grad)<10 else params.grad[:10])
        # feature : model(body) output
        x = feature[:, 0, :]
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layernorm1(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layernorm2(x)
        x = self.dropout(x)

        x = self.projection(x)
        return x

# BCELoss
class TwoSentenceClassificationHead(nn.Module):
    """
    for rte, mrpc, qqp, mnli
    output : [bs, 1]
    """
    def __init__(self,input_dim, inner_dim, pooler_dropout, device):
        """

        input_dim: hidden_dim * 2 (!!! be careful !!!  two [cls] tokens will be concatenated)
        inner_dim: hidden_dim

        """
        super().__init__()
        self.ffn = nn.Sequential(nn.Tanh(),nn.Linear(input_dim, inner_dim), nn.ReLU(),nn.Linear(inner_dim,1)).to(device)
        self.layer1 = nn.Linear(input_dim, inner_dim).to(device)
        self.layer2 = nn.Linear(inner_dim, inner_dim).to(device)
        self.projection = nn.Linear(inner_dim, 1).to(device)
        self.relu = nn.ReLU().to(device)
        self.layernorm1 = nn.BatchNorm1d(inner_dim).to(device)
        self.layernorm2 = nn.BatchNorm1d(inner_dim).to(device)
        self.dropout = nn.Dropout(p=pooler_dropout)
    def forward(self, feature, second_cls_idx):
        # indices = torch.tensor([0, second_cls_idx])  # indices for selecting 1st and 2nd [cls]
        cls1 = feature[:,0,:] # first [cls]

        cls2 = feature[second_cls_idx[0],second_cls_idx[1]]  # access two [cls] tokens embedding
        x = torch.cat([cls1, cls2], dim=-1)
        # x : [bs , 2 ,hidden_dim]

        x = x.view(x.shape[0], -1)
        # concat two [cls] token embedding
        # x: [bs, hidden_dim * 2]
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layernorm1(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layernorm2(x)
        x = self.dropout(x)
        x = self.projection(x)

        return x
# CrossEntropyLoss
class TwoSentenceRegressionHead(nn.Module):
    """
    for stsb
    output : [1] (cosine_similarity * 5)
    """
    def __init__(self,input_dim, inner_dim, pooler_dropout, cos_eps, device):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim).to(device)
        self.activation_fn = nn.GELU().to(device)
        self.dropout = nn.Dropout(p=pooler_dropout).to(device)
        self.out_proj = nn.Linear(inner_dim, 2).to(device)
        self.cos = nn.CosineSimilarity(dim=1, eps=cos_eps)

    def forward(self, feature, second_cls_idx):
        cls1 = feature[:,0,:] # first [cls]
        cls2 = feature[second_cls_idx[0], second_cls_idx[1]] # second [cls]
        # get cosine similarity between two [cls] tokens
        similarity = self.cos(cls1, cls2)

        # return with multiplying 5
        return similarity


class gMLP_ClassificationModel(gMLP_LanguageModel):
    def __init__(self,vocab_size, d_model, d_ffn, seq_len, num_layers,device,output_logits=False,task_type="one"):
        super().__init__(vocab_size, d_model, d_ffn, seq_len, num_layers,device,False)
        self.device = device
        self.OneSentence = OneSentenceClassificationHead(d_model, int(d_model/2), 0.15, device)
        self.TwoSentence = TwoSentenceClassificationHead(d_model*2, d_model, 0.15, device)
        # "one" => cola, sst2
        # "two" => stsb, rte, mrpc, qqp, mnli
        self.task_type = task_type

    def forward(self,x, token_type_ids):
        embedding = self.embed(x, token_type_ids)
        embedding = embedding.to(self.device)
        output = self.model(embedding)
        if self.task_type == "one":
            output = self.OneSentence(output)
        # pass second cls index to head
        if self.task_type == "two":
            indices = (x==101).nonzero()
            indices = indices.transpose(0,1)
            indices = indices[:,indices[1]>0]
            output = self.TwoSentence(output, indices)

        return output

class gMLP_RegressionModel(gMLP_LanguageModel):
    def __init__(self,vocab_size, d_model, d_ffn, seq_len, num_layers,device,output_logits=False,task_type="one"):
        super().__init__(vocab_size, d_model, d_ffn, seq_len, num_layers,device,False)
        self.device = device
        # stsb
        self.Regression = TwoSentenceRegressionHead(d_model, int(d_model/2), 0.15, 1e-8, self.device)
        

    def forward(self,x, token_type_ids):
        embedding = self.embed(x, token_type_ids)
        embedding = embedding.to(self.device)
        output = self.model(embedding)
        output = self.Regression(output)

        return output

"""
tmp_model = build_model(28996,256,1024,256,4)

params = list(tmp_model.parameters())
print("The number of parameters:",sum([p.numel() for p in tmp_model.parameters() if p.requires_grad]), "elements")

The number of parameters: 18580300 elements
"""
def build_model(num_tokens, d_model, d_ffn, seq_len, num_layers,device):
    
    model = gMLP_LanguageModel(num_tokens,d_model,d_ffn,
                            seq_len,num_layers,device,False).to(device)
    
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

def build_regression_model(num_tokens, d_model, d_ffn, seq_len, num_layers,device):
    model = gMLP_RegressionModel(num_tokens,d_model,d_ffn,
                            seq_len,num_layers,device,False).to(device)
    
    if torch.cuda.device_count()>1:
        print("Using ",torch.cuda.device_count(),"GPUs in total!")
        model = torch.nn.DataParallel(model,device_ids=[0,1,2,3],output_device=1)
    
    return model.cuda() if torch.cuda.is_available() else model