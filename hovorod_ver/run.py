import logging
import sys
from horovod_ver.config.configs import get_config
from horovod_ver.src.pretrain import Pretrain_Trainer
#from src.finetune import Finetune_Trainer

def main(parser, usage_mode):
    # TO BE UPDATED
    supported_tasks = ['classification','summarization']
    
    # pretrain
    if usage_mode == 'pretrain':
        te = open("pretrain_log.txt",'w')
        class BufferHandler:
            def __init__(self, stream):
                self.stream = stream
            
            def write(self, data):
                self.stream.write(data)
                self.stream.flush()
                te.write(data)
        sys.stdout = BufferHandler(sys.stdout)
        sys.stdout.write('#################################################\n')
        sys.stdout.write('You have entered PreTrain mode.\n')
        sys.stdout.write('#################################################\n')

        # train with Pretrain_Trainer
        trainer = Pretrain_Trainer(parser)
        trainer.train_test()

        sys.stdout.write('#################################################\n')
        sys.stdout.write('PreTrain mode has ended.\n')
        sys.stdout.write('#################################################\n')

        sys.stdout.close()
    # finetune
    elif usage_mode == 'finetune':
        task = input('Which dataset are you finetuning on :')
        te = open(f"finetune_{task}_log.txt",'w')
        class BufferHandler:
            def __init__(self, stream):
                self.stream = stream
            
            def write(self, data):
                self.stream.write(data)
                self.stream.flush()
                te.write(data)
        sys.stdout = BufferHandler(sys.stdout)
        sys.stdout.write('#################################################\n')
        sys.stdout.write('You have entered Finetune mode.\n')
        sys.stdout.write('#################################################\n')

        # train with Finetune_Trainer
        trainer = Finetune_Trainer(parser)
        trainer.train_test()

        sys.stdout.write('#################################################\n')
        sys.stdout.write('Finetune mode has ended.\n')
        sys.stdout.write('#################################################\n')
        
        sys.stdout.close()
    # inference
    elif usage_mode in supported_tasks:
        assert "Not supported yet!"
    
    else:
        assert "You have gave wrong mode"

if __name__ == "__main__":
    sys.stdout.write('#################################################\n')
    sys.stdout.write('You have entered __main__.\n')
    sys.stdout.write('#################################################\n')
    
    # define ArgumentParser
    parser = get_config()

    # get user input of whether purpose is train or inference
    usage_mode = input('Enter the mode you want to use :')

    # run main
    main(parser, usage_mode)

    sys.stdout.write('#################################################\n')
    sys.stdout.write('You are exiting __main__.\n')
    sys.stdout.write('#################################################\n')
