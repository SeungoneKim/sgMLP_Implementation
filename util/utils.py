import json

import scipy.stats
import sklearn.metrics as skm
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_metric
from models.model import build_model
from config.configs import get_config
from nltk.translate.bleu_score import sentence_bleu

huggingface_metrics_list = ['bertscore','bleu','bleurt','coval','gleu','glue','meteor',
                            'rouge','sacrebleu','seqeval','squad','squad_v2','xlni']

sklearn_metrics_list = ['accuracy_score','f1_score','precision_score','recall_score',
                        'roc_auc_score','mean_squared_error','mean_absolute_error']

lossfn_list = ['BCELoss','CrossEntropyLoss','KLDivLoss','BCEWithLogitsLoss',
                'L1Loss','MSELoss','NLLLoss','MAE']

def load_metricfn(metric_type):
    metric= None
    if metric_type == 'bleu':
        metric = sentence_bleu
    if metric_type == 'pearson':
        metric = scipy.stats.pearsonr
    elif metric_type in huggingface_metrics_list:
        metric= load_metric(metric_type)
    elif metric_type in sklearn_metrics_list:
        if metric_type == 'accuracy_score':
            metric = skm.accuracy_score
        elif metric_type == 'f1_score':
            metric = skm.f1_score
        elif metric_type == 'precision_score':
            metric = skm.precision_score
        elif metric_type == 'recall_score':
            metric = skm.recall_score
        elif metric_type == 'roc_auc_score':
            metric = skm.roc_auc_score
        elif metric_type == 'mean_squared_error':
            metric = skm.mean_squared_error
        elif metric_type == 'mean_absolute_error':
            metric = skm.mean_absolute_error
    else:
        assert "You typed a metric that doesn't exist or is not supported"

    return metric

def load_lossfn(lossfn_type,device,ignore_idx=None):
    lossfn= None
    if lossfn_type in lossfn_list:
        if lossfn_type == 'BCELoss': #
            lossfn = nn.BCELoss()
        elif lossfn_type == 'CrossEntropyLoss':
            if ignore_idx != None:
                lossfn = nn.BCEWithLogitsLoss(ignore_index=ignore_idx).to(device) # 알고보니 BCELoss였던거....;;
            else:
                lossfn = nn.BCEWithLogitsLoss().to(device)
        elif lossfn_type == 'KLDivLoss':
            lossfn = nn.KLDivLoss()
        elif lossfn_type == 'BCEWithLogitsLoss':
            lossfn = nn.BCEWithLogitsLoss()
        elif lossfn_type == 'L1Loss':
            lossfn = nn.L1Loss()
        elif lossfn_type == 'MSELoss':
            lossfn = nn.MSELoss()
        elif lossfn_type == 'NLLLoss':
            if ignore_idx != None:
                lossfn = nn.NLLLoss(ignore_index=ignore_idx).to(device)
            else:
                lossfn = nn.NLLLoss().to(device)
    
    return lossfn

# Adam Optimizer
def load_optimizer(model, learning_rate, weight_decay, beta1, beta2):
    return optim.AdamW(params=model.parameters(), lr=learning_rate, 
                        weight_decay=weight_decay, betas=[beta1, beta2])

# Linear Scheduler with WarmUp
def load_scheduler(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1,num_warmup_steps))

    return optim.lr_scheduler.LambdaLR(optimizer= optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() >1:
        nn.init.kaiming_uniform_(m.weight.data)

# Convert index into word
def convert_idx_to_word(idx_sent, vocab):
    words=[]
    # convert using vocabulary
    for idx_token in idx_sent:
        word_token = vocab[idx_token]
        if "<" not in word_token:
            words.append(word_token)
    
    words = " ".join(words)
    
    return words

# Measure time spent
def time_measurement(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    
    return elapsed_mins, elapsed_secs

# Count the number of parameters the model has
def count_parameters(model):
    params = list(model.parameters())
    print("The number of parameters:",sum([p.numel() for p in model.parameters() if p.requires_grad]), "elements")

def save_checkpoint(model, optimizer, epoch, save_path):
    # saving state_dict of model and optimizer, and also saving epoch info
    torch.save({
        'model_state_dict': model.state_dict(),
        'mlm_optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    },save_path)

def save_bestmodel(model, optimizer, parser, save_path):
    # saving state_dict of model and optimizer
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    },save_path)

    # saving model information as txt file
    model_info = {}
    model_info["model_size"] = []
    for param_tensor in model.state_dict():
        model_info["model_size"][param_tensor] = str(model.state_dict()[param_tensor].size())    

    with open('model_information.txt','w') as f:
        json.dump(model_info,f,indent=2)

def load_checkpoint(model, load_path):
    # loading state_dict of model, and also loading epoch info
    pretrained_dict = torch.load(load_path,map_location=torch.device('cuda'))['model_state_dict']
    
    model_dict = model.state_dict()

    pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)

    model.load_state_dict(model_dict)

    return model