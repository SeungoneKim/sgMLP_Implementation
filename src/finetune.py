import os
import sys
import argparse
import logging
from tqdm.notebook import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import shutil
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import transformers
from config.configs import set_random_fixed, get_path_info
from data.dataloader import get_Finetune_dataloader_Atype, get_Finetune_dataloader_Btype
from data.tokenizer import Tokenizer
from util.utils import (load_metricfn, load_optimizer, load_scheduler, load_lossfn, 
                        save_checkpoint, load_checkpoint,
                        time_measurement, count_parameters, initialize_weights)
from util.optim_scheduler import ScheduledOptim
from models.model import build_classification_model, build_regression_model
import wandb

class Finetune_Trainer():
    def __init__(self, parser, task):
        
        # set parser
        self.args = parser.parse_args()
        #initialize wandb
        #wandb.init(name=task)
        # save loss history to plot later on
        self.training_history = []
        self.validation_history = []

        # set variables needed for training
        self.n_epoch = self.args.epoch
        self.train_batch_size = self.args.train_batch_size
        self.display_step = self.args.display_step # training
        self.val_batch_size = self.args.val_batch_size
        self.test_batch_size = self.args.test_batch_size
        self.display_examples = self.args.display_examples # testing
        
        self.lr = self.args.init_lr
        #self.eps = self.args.adam_eps
        self.weight_decay = self.args.weight_decay
        self.beta1 = self.args.adam_beta1
        self.beta2 = self.args.adam_beta2

        self.warmup_steps = self.args.warm_up
        #self.factor = self.args.factor
        #self.patience = self.args.patience
        #self.clip = self.args.clip

        self.language = self.args.language
        self.max_len = self.args.max_len
        self.vocab_size = self.args.vocab_size

        self.device = self.args.device
        self.pretrain_weightpath = os.path.join(os.getcwd(),'weights')
        if os.path.isdir('finetune_weights'):
            shutil.rmtree("finetune_weights")
        self.weightpath = os.path.join(os.getcwd(),'finetune_weights')
        self.final_weightpath = os.path.join(os.getcwd(),'final_finetune_weights')

        self.best_pretrain_epoch = self.args.best_pretrain_epoch
        
        # build dataloader
        self.task = task
        task_Atype = ['cola','sst2']
        task_Btype = ['stsb','rte','mrpc','qqp','mnli']
        self.task_Btype = ['stsb','rte','mrpc','qqp','mnli']
        task_Btype_sentence = ['stsb','rte','mrpc']
        task_Btype_question = ['qqp']
        task_Btype_hypothesis = ['mnli']
        if task in task_Atype:
            self.train_dataloader, self.val_dataloader, self.test_dataloader = get_Finetune_dataloader_Atype(
                self.train_batch_size, self.val_batch_size, self.test_batch_size,
                self.language, self.max_len,
                'glue', task, 'sentence', 'label',
                None
            )
        elif task in task_Btype_sentence:
            self.train_dataloader, self.val_dataloader, self.test_dataloader = get_Finetune_dataloader_Btype(
                self.train_batch_size, self.val_batch_size, self.test_batch_size,
                self.language, self.max_len,
                self.args.dataset_name, self.args.dataset_type, 'sentence1', 'sentence2', 'label',
                None
            )
        elif task in task_Btype_question:
            self.train_dataloader, self.val_dataloader, self.test_dataloader = get_Finetune_dataloader_Btype(
                self.train_batch_size, self.val_batch_size, self.test_batch_size,
                self.language, self.max_len,
                self.args.dataset_name, self.args.dataset_type, 'question1', 'question2', 'label',
                None
            )
        elif task in task_Btype_hypothesis:
            self.train_dataloader, self.val_dataloader, self.test_dataloader = get_Finetune_dataloader_Btype(
                self.train_batch_size, self.val_batch_size, self.test_batch_size,
                self.language, self.max_len,
                self.args.dataset_name, self.args.dataset_type, 'premise', 'hypothesis', 'label',
                None
            )
        else:
            assert "The task you typed in is not supported!"

        self.train_batch_num = len(self.train_dataloader)
        self.val_batch_num = len(self.val_dataloader)
        self.test_batch_num = len(self.test_dataloader)
        
        self.num_training_steps = (self.train_batch_num) * (self.n_epoch)

        self.t_total = self.train_batch_num * self.n_epoch

        # load metric
        if task == 'mnli':
            self.metric = load_metricfn('matthews_corrcoef')
        elif task == 'stsb':
            self.metric = load_metricfn('pearson')
        else:
            self.metric = load_metricfn('accuracy_score')
        
        # build model
        if task in task_Atype:
            self.model= build_classification_model(self.vocab_size, self.args.model_dim, self.args.hidden_dim, 
                                                   self.max_len, self.args.num_layers, self.device, 'one')
        elif task == 'stsb':
            self.model = build_regression_model(self.vocab_size, self.args.model_dim, self.args.hidden_dim, 
                                                   self.max_len, self.args.num_layers, self.device)
        else:
            self.model = build_classification_model(self.vocab_size, self.args.model_dim, self.args.hidden_dim, 
                                                   self.max_len, self.args.num_layers, self.device, 'two')
        
        
        load_checkpoint(self.model, os.path.join(self.pretrain_weightpath,str(self.best_pretrain_epoch)+".pth"))

        # build optimizer
        self.optimizer = load_optimizer(self.model, self.lr, self.weight_decay, 
                                        self.beta1, self.beta2)
        
        # build scheduler
        self.optim_scheduler = ScheduledOptim(self.optimizer, self.args.model_dim, self.warmup_steps)
        
        # build lossfn
        if task=='stsb':
            self.lossfn = load_lossfn('MSELoss',self.args.pad_idx)              # Regression
        else:
            self.lossfn = load_lossfn('CrossEntropyLoss',self.args.pad_idx)     # Classification

    def train_test(self):
        best_model_epoch, training_history, validation_history = self.finetune()
        self.test(best_model_epoch)
        self.plot(training_history, validation_history)

    def finetune(self):
        # set logging        
        logging.basicConfig(level=logging.WARNING)
        
        # logging message
        sys.stdout.write('#################################################\n')
        sys.stdout.write('You have started training the model.\n')
        print('Your model size is : ')
        count_parameters(self.model)
        sys.stdout.write('#################################################\n')

        # set randomness of training procedure fixed
        self.set_random(516)
        
        # build directory to save to model's weights
        self.build_directory()

        # set initial variables for training, validation
        train_batch_num = len(self.train_dataloader)
        validation_batch_num = len(self.val_dataloader)

        # set initial variables for model selection
        best_model_epoch=0
        best_model_score=0
        best_model_loss =float('inf')

        # save information of the procedure of training
        training_history=[]
        validation_history=[]

        # predict when training will end based on average time
        total_time_spent = 0
        
        # start of looping through training data
        for epoch_idx in range(self.n_epoch):
            # measure time when epoch start
            start_time = time.time()
            
            sys.stdout.write('#################################################\n')
            sys.stdout.write(f"Epoch : {epoch_idx+1} / {self.n_epoch}")
            sys.stdout.write('\n')
            sys.stdout.write('#################################################\n')

            ########################
            #### Training Phase ####
            ########################
            
            # switch model to train mode
            self.model.train()

            # set initial variables for training (inside epoch)
            training_loss_per_epoch=0.0
            training_acc_per_epoch = 0

            # train model using batch gradient descent with Adam Optimizer
            for batch_idx, batch in tqdm(enumerate(self.train_dataloader)):
                # move batch of data to gpu
                input_ids = batch['input_ids']                      #[bs, 1, sl]
                token_type_ids = batch['token_type_ids']            #[bs, 1, sl]
                labels = batch['label'].to(torch.float)                             #[bs, 1]

                # reshape input_ids and token_type_ids
                if self.task in self.task_Btype:
                    reshaped_input_ids = input_ids.to(self.device)
                    reshaped_token_type_ids = token_type_ids.contiguous().cuda(reshaped_input_ids.device)

                else:
                    reshaped_input_ids = input_ids.contiguous().permute(0,2,1).squeeze(2).to(self.device)
                    reshaped_token_type_ids = token_type_ids.contiguous().permute(0,2,1).squeeze(2).cuda(reshaped_input_ids.device)
                # reshape input_ids and token_type_ids
                reshaped_labels = labels.contiguous().squeeze(1).cuda(reshaped_input_ids.device)

                # compute model output
                # 1 sentence classification : Cola, SST2
                # 2 sentence classification : RTE, MRPC, QQP, MNLI
                # 2 sentence regression : STSB
                model_output = self.model(reshaped_input_ids, reshaped_token_type_ids).squeeze()          # [bs, 2] in classification, [bs, 1] in regression
                train_pred = torch.tensor([1 if n >0 else 0 for n in model_output]).to(self.device)
                training_acc_per_epoch += self.metric(train_pred.cpu().detach().numpy(), reshaped_labels.cpu().detach().numpy())


                # print(model_output.float().type())
                # print(model_output)
                # print(reshaped_labels.type())
                # print(reshaped_labels)
                if batch_idx == 0:
                    print("##### train pred #####")
                    print(model_output)
                    print(reshaped_labels)
                    print("#"*len("##### train pred #####"))
                # compute loss using model output and labels(reshaped ver)
                loss = self.lossfn(model_output, reshaped_labels)

                # clear gradients, and compute gradient with current batch
                self.optimizer.zero_grad()
                loss.backward()

                # clip gradients
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.clip)

                # update gradients
                self.optim_scheduler.step_and_update_lr()

                # add loss to training_loss
                training_loss_per_iteration = loss.item()
                training_loss_per_epoch += training_loss_per_iteration

                # Display summaries of training procedure with period of display_step
                if ((batch_idx+1) % self.display_step==0) and (batch_idx>0):
                    sys.stdout.write(f"Training Phase |  Epoch: {epoch_idx+1} |  Step: {batch_idx+1} / {train_batch_num} | loss : {training_loss_per_iteration}")
                    sys.stdout.write('\n')

            # save training loss of each epoch, in other words, the average of every batch in the current epoch
            training_mean_loss_per_epoch = training_loss_per_epoch / train_batch_num
            training_history.append(training_mean_loss_per_epoch)
            training_acc_per_epoch = (training_acc_per_epoch/train_batch_num)*100

            ##########################
            #### Validation Phase ####
            ##########################

            # switch model to eval mode
            self.model.eval()

            # set initial variables for validation (inside epoch)
            validation_loss_per_epoch=0.0 
            validation_score_per_epoch=0.0

            # validate model using batch gradient descent with Adam Optimizer
            for batch_idx, batch in tqdm(enumerate(self.val_dataloader)):
                # move batch of data to gpu
                input_ids = batch['input_ids']                      #[bs, 1, sl]
                token_type_ids = batch['token_type_ids']            #[bs, 1, sl]
                labels = batch['label'].to(torch.float)                             #[bs, 1]

               # reshape input_ids and token_type_ids
                if self.task in self.task_Btype:
                    reshaped_input_ids = input_ids.to(self.device)
                    reshaped_token_type_ids = token_type_ids.contiguous().cuda(reshaped_input_ids.device)

                else:
                    reshaped_input_ids = input_ids.contiguous().permute(0,2,1).squeeze(2).to(self.device)
                    reshaped_token_type_ids = token_type_ids.contiguous().permute(0,2,1).squeeze(2).cuda(reshaped_input_ids.device)

                reshaped_labels = labels.contiguous().squeeze(1).cuda(reshaped_input_ids.device)


                # compute model output
                # 1 sentence classification : Cola, SST2
                # 2 sentence classification : RTE, MRPC, QQP, MNLI
                # 2 sentence regression : STSB
                with torch.no_grad():
                    model_output = self.model(reshaped_input_ids, reshaped_token_type_ids).squeeze()            # [bs, 2] in classification, [bs, 1] in regression

                if batch_idx == 0:

                    print(model_output)
                    print(reshaped_labels)
                # compute loss using model output and labels(reshaped ver)
                loss = self.lossfn(model_output, reshaped_labels)

                # add loss to training_loss
                validation_loss_per_iteration = loss.item()
                validation_loss_per_epoch += validation_loss_per_iteration

                # reshape model output
                reshaped_model_output = torch.tensor([1 if n >0 else 0 for n in model_output.squeeze()]).to(self.device)

                # compute bleu score using model output and labels(reshaped ver)
                validation_score_per_iteration = self.metric(reshaped_model_output.cpu().detach().numpy(), reshaped_labels.cpu().detach().numpy())*100
                validation_score_per_epoch += validation_score_per_iteration

            # save validation loss of each epoch, in other words, the average of every batch in the current epoch
            validation_mean_loss_per_epoch = validation_loss_per_epoch / validation_batch_num
            validation_history.append(validation_mean_loss_per_epoch)

            # save validation score of each epoch, in other words, the average of every batch in the current epoch
            validation_mean_score_per_epoch = validation_score_per_epoch / validation_batch_num

            # Display summaries of validation result after all validation is done
            sys.stdout.write(f"Validation Phase |  Epoch: {epoch_idx+1} | loss : {validation_mean_loss_per_epoch} | score : {validation_mean_score_per_epoch}")
            sys.stdout.write('\n')

            # Model Selection Process using validation_mean_score_per_epoch
            if (validation_mean_loss_per_epoch < best_model_loss):
                best_model_epoch = epoch_idx+1
                best_model_loss = validation_mean_loss_per_epoch
                best_model_score = validation_mean_score_per_epoch

                save_checkpoint(self.model, self.optimizer, epoch_idx,
                            os.path.join(self.weightpath,str(epoch_idx+1)+".pth"))

            #wandb log
            train_log_dict = {
            "train/step": epoch_idx,  # grows exponentially with internal wandb step
            "train/loss": training_mean_loss_per_epoch, # x-axis is train/step
            "train/accuracy": training_acc_per_epoch} # x-axis is train/step
            val_log_dict ={
            "val/loss": validation_mean_loss_per_epoch, # x-axis is internal wandb step
            "val/accuracy":validation_mean_score_per_epoch

           }
            # wandb.log(train_log_dict)
            # wandb.log(val_log_dict)
            # measure time when epoch end
            end_time = time.time()

            # measure the amount of time spent in this epoch
            epoch_mins, epoch_secs = time_measurement(start_time, end_time)
            sys.stdout.write(f"Time spent in epoch {epoch_idx+1} is {epoch_mins} minuites and {epoch_secs} seconds\n")
            
            # measure the total amount of time spent until now
            total_time_spent += (end_time - start_time)
            total_time_spent_mins = int(total_time_spent/60)
            total_time_spent_secs = int(total_time_spent - (total_time_spent_mins*60))
            sys.stdout.write(f"Total amount of time spent until epoch {epoch_idx+1} is {total_time_spent_mins} minuites and {total_time_spent_secs} seconds\n")

            # calculate how more time is estimated to be used for training
            #avg_time_spent_secs = total_time_spent_secs / (epoch_idx+1)
            #left_epochs = self.n_epoch - (epoch_idx+1)
            #estimated_left_time = avg_time_spent_secs * left_epochs
            #estimated_left_time_mins = int(estimated_left_time/60)
            #estimated_left_time_secs = int(estimated_left_time - (estimated_left_time_mins*60))
            #sys.stdout.write(f"Estimated amount of time until epoch {self.n_epoch} is {estimated_left_time_mins} minuites and {estimated_left_time_secs} seconds\n")

        # summary of whole procedure    
        sys.stdout.write('#################################################\n')
        sys.stdout.write(f"Training and Validation has ended.\n")
        sys.stdout.write(f"Your best model was the model from epoch {best_model_epoch+1} and scored {self.args.metric} score : {best_model_score} | loss : {best_model_loss}\n")
        sys.stdout.write('#################################################\n')

        return best_model_epoch, training_history, validation_history
    

    def test(self, best_model_epoch):

        # logging message
        sys.stdout.write('#################################################\n')
        sys.stdout.write('You have started testing the model.\n')
        sys.stdout.write('#################################################\n')

        # set randomness of training procedure fixed
        self.set_random(516)
        
        # build directory to save to model's weights
        self.build_final_directory()

        # loading the best_model from checkpoint
        task_Atype = ['cola','sst2']
        if self.task in task_Atype:
            best_model= build_classification_model(self.vocab_size, self.args.model_dim, self.args.hidden_dim, 
                                                   self.max_len, self.args.num_layers, self.device, 'one')
        elif self.task == 'stsb':
            best_model = build_regression_model(self.vocab_size, self.args.model_dim, self.args.hidden_dim, 
                                                   self.max_len, self.args.num_layers, self.device)
        else:
            best_model = build_classification_model(self.vocab_size, self.args.model_dim, self.args.hidden_dim, 
                                                   self.max_len, self.args.num_layers, self.device, 'two')
        
        load_checkpoint(best_model,
                    os.path.join(self.weightpath,str(best_model_epoch)+".pth"))

        # set initial variables for test
        test_batch_num = len(self.test_dataloader)

        ##########################
        ######  Test Phase  ######
        ##########################

        # switch model to eval mode
        best_model.eval()

        # set initial variables for validation (inside epoch)
        test_score_per_epoch=0.0
        test_score_tmp_list=[]

        # validate model using batch gradient descent with Adam Optimizer
        for batch_idx, batch in tqdm(enumerate(self.test_dataloader)):
            # move batch of data to gpu
            input_ids = batch['input_ids']                      #[bs, 1, sl]
            token_type_ids = batch['token_type_ids']            #[bs, 1, sl]
            labels = batch['label']                             #[bs, 1]

            # reshape input_ids and token_type_ids
            reshaped_input_ids = input_ids.contiguous().permute(0,2,1).squeeze(2).to(self.device)
            reshaped_token_type_ids = token_type_ids.contiguous().permute(0,2,1).squeeze(2).cuda(reshaped_input_ids.device)
            reshaped_labels = labels.contiguous().squeeze(1).cuda(reshaped_input_ids.device)

            # compute model output
            # 1 sentence classification : Cola, SST2
            # 2 sentence classification : RTE, MRPC, QQP, MNLI
            # 2 sentence regression : STSB
            with torch.no_grad():
                model_output = self.model(reshaped_input_ids, reshaped_token_type_ids)            # [bs, 2] in classification, [bs, 1] in regression

            # reshape model output
            reshaped_model_output = model_output.argmax(dim=1)
            
            # compute bleu score using model output and labels(reshaped ver)
            test_score_per_iteration = self.metric(reshaped_model_output.cpu().detach().numpy(), reshaped_labels.cpu().detach().numpy())*100
            test_score_tmp_list.append(test_score_per_iteration)
            test_score_per_epoch += test_score_per_iteration

        # calculate test score        
        test_score_per_epoch = test_score_per_epoch / test_batch_num

        # Evaluate summaries with period of display_steps
        sys.stdout.write(f"Test Phase |  Best Epoch: {best_model_epoch} | score : {test_score_per_epoch}\n")

        save_checkpoint(self.model, self.optimizer, 1,
                            os.path.join(self.final_weightpath,"final_"+self.task+".pth"))



    def plot(self, training_history, validation_history):
        step = np.linspace(0,self.n_epoch,self.n_epoch)
        plt.plot(step,np.array(training_history),label='Training')
        plt.plot(step,np.array(validation_history),label='Validation')
        plt.xlabel('number of epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        cur_path = os.getcwd()
        plt.savefig(cur_path)

        sys.stdout.write('Image of train, validation history saved as plot png!\n')

    def build_directory(self):
        # Making directory to store model pth
        curpath = os.getcwd()
        weightpath = os.path.join(curpath,'finetune_weights')
        os.mkdir(weightpath)

    def build_final_directory(self):
        curpath = os.getcwd()
        final_weightpath = os.path.join(curpath,'final_finetune_weights')
        os.mkdir(final_weightpath)

    def set_random(self, seed_num):
        set_random_fixed(seed_num)

