import os
import sys
import argparse
import random
import torch
import numpy as np

parser = argparse.ArgumentParser()
if torch.cuda.device_count()>1:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
# gpu
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else "cpu")
# hyperparameters
parser.add_argument('--epoch', type=int, default=1)
parser.add_argument('--train_batch_size', type=int, default=512)
parser.add_argument('--display_step',type=int, default=100)
parser.add_argument('--val_batch_size',type=int, default=256)
parser.add_argument('--test_batch_size',type=int,default=2)
parser.add_argument('--display_examples',type=int, default=1000)

parser.add_argument('--model_dim', type=int, default=512)
parser.add_argument('--key_dim',type=int, default = 512)
parser.add_argument('--value_dim',type=int, default= 512)
parser.add_argument('--hidden_dim', type=int, default=2048)
parser.add_argument('--num_layers', type=int, default=12)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--drop_prob',type=float, default=0.1)

parser.add_argument('--init_lr',type=float, default=2e-5)
parser.add_argument('--warm_up',type=int, default=10000)
#parser.add_argument('--adam_eps',type=float, default=5e-9)
parser.add_argument('--adam_beta1',type=float, default=0.9)
parser.add_argument('--adam_beta2',type=float, default=0.999)
#parser.add_argument('--patience',type=int,default=10)
#parser.add_argument('--factor',type=float,default=0.9)
parser.add_argument('--clip',type=int, default=1)
parser.add_argument('--weight_decay',type=float, default=0.01)
parser.add_argument('--decay_epoch',type=int, default=10000)

parser.add_argument('--max_len', type=int, default=64)
parser.add_argument('--vocab_size',type=int, default=28996)

# tokenizer
parser.add_argument('--language',type=str, default='en')
parser.add_argument('--pad_idx',type=int, default=0) # [PAD]
parser.add_argument('--unk_idx',type=int, default=100) # [UNK]
parser.add_argument('--cls_idx',type=int, default=101)  # [CLS]
parser.add_argument('--sep_idx',type=int, default=102)  # [SEP]
parser.add_argument('--mask_idx',type=int,default=103) # [MASK]
# trainer
parser.add_argument('--metric',type=str, default='accuracy_score') # For Finetuning
parser.add_argument('--pretrain_lossfn',type=str, default= 'NLLLoss')
parser.add_argument('--finetune_lossfn',type=str, default= 'CrossEntropyLoss')
# dataloader
# pretrain
parser.add_argument('--pretrain_dataset_name',type=str, default='bookcorpus') # bookcorpus
parser.add_argument('--pretrain_dataset_type',type=str, default='plain_text') # plain_text
parser.add_argument('--pretrain_category_name',type=str, default='text') # text
parser.add_argument('--pretrain_strategy',type=str, default='MLM') # MLM
parser.add_argument('--pretrain_percentage',type=int, default=100)

parser.add_argument('--pretrain_next_sent_prob',type=float,default=0.5)
parser.add_argument('--pretrain_masking_prob',type=float,default=0.15)

parser.add_argument('--pretrain_training_ratio',type=float,default=0.8)
parser.add_argument('--pretrain_validation_ratio',type=float,default=0.1)
parser.add_argument('--pretrain_test_ratio',type=float,default=0.1)
# finetune
parser.add_argument('--finetune_dataset_name',type=str, default=None) # wmt14
parser.add_argument('--finetune_dataset_type',type=str, default=None) # de-en
parser.add_argument('--finetune_category_name',type=str, default=None) # translation
parser.add_argument('--finetune_x_name',type=str, default=None) # de
parser.add_argument('--finetune_y_name',type=str, default=None) # en
parser.add_argument('--finetune_percentage',type=int, default=100)

parser.add_argument('--model_path', type=str, default=None)

def get_config():
    return parser

def set_random_fixed(seed_num):
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)

    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

    np.random.seed(seed_num)

def get_path_info():
    cur_path = os.getcwd()
    weight_path = os.path.join(cur_path,'weights')
    final_model_path = os.path.join(cur_path,'final_results')
    
    return cur_path, weight_path, final_model_path