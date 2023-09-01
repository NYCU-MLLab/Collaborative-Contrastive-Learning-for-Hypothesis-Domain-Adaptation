'''
This is the main code of the ECAPATDNN project, to define the parameters and build the construction
'''

import argparse, glob, os, torch, warnings, time, logging
from tools import *
import numpy as np
from dataLoader import data_loader
from CHDAModel import CHDAModel
from torch.utils.data import WeightedRandomSampler
from itertools import cycle
from tqdm import tqdm


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description = "ECAPA_trainer")
## Training Settings
parser.add_argument('--num_frames',         type=int,   default=200,     help='Duration of the input segments, eg: 200 for 2 second')
parser.add_argument('--max_epoch',          type=int,   default=25,      help='Maximum number of epochs')
parser.add_argument('--batch_size',         type=int,   default=128,      help='Batch size')
parser.add_argument('--n_cpu',              type=int,   default=4,       help='Number of loader threads')
parser.add_argument('--test_step',          type=int,   default=1,       help='Test and save every [test_step] epochs')
parser.add_argument('--lr',                 type=float, default=0.001,   help='Learning rate')
parser.add_argument("--lr_decay",           type=float, default=0.9,     help='Learning rate decay every [test_step] epochs')

## Training and evaluation path/lists, save path
parser.add_argument('--dataset',  		type=str,   required=True)

### Train CNCeleb1
parser.add_argument('--train_list', 	type=str,   default="/home/sv/CN-Celeb_flac/train.csv",    				   help='The path of the CNCeleb training list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt')
parser.add_argument('--train_path', 	type=str,   default="/home/sv/CN-Celeb_flac/data",  					   help='The path of the CNCeleb training data')
##  Eval CNCeleb1
parser.add_argument('--eval_list',  	type=str,   default="/home/sv/CN-Celeb_flac/eval/lists/trials.lst",        help='The path of the CNCeleb evaluation list')
parser.add_argument('--eval_path',  	type=str,   default="/home/sv/CN-Celeb_flac/eval",                    	   help='The path of the CNCeleb evaluation data')

### Train CommonVoice
#parser.add_argument('--train_list', 	type=str,   default="/home/sv/cv-corpus-13.0-2023-03-09/zh-TW/train.tsv",  help='The path of the Common Voice training list')
#parser.add_argument('--train_path', 	type=str,   default="/home/sv/cv-corpus-13.0-2023-03-09/zh-TW/clips",      help='The path of the Common Voice  training data')
### Eval CommonVoice
#parser.add_argument('--eval_list',      type=str,   default="/home/sv/cv-corpus-13.0-2023-03-09/zh-TW/eval.tsv",  help='The path of the Common Voice evaluation list')
#parser.add_argument('--eval_path',  	type=str,   default="/home/sv/cv-corpus-13.0-2023-03-09/zh-TW/clips",      help='The path of the Common Voice evaluation data')

parser.add_argument('--musan_path', 	type=str,   default="/home/sv/musan",  													 	help='The path to the MUSAN set, eg:"/musan_split" in my case')
parser.add_argument('--rir_path',   	type=str,   default="/home/sv/voxceleb_trainer/data2/voxceleb2/RIRS_NOISES/simulated_rirs", help='The path to the RIR set, eg:"/simulated_rirs" in my case');
parser.add_argument('--save_path',  	type=str,   default="",    													                help='Path to save the score.txt and models')
parser.add_argument('--initial_model',  type=str,   default="",  																	help='Path of the initial_model')

## Model and Loss settings
parser.add_argument('--C',              type=int,   default=1024,   				help='Channel size for the speaker encoder')
parser.add_argument('--m',              type=float, default=0.25,    				help='Loss margin in AAM softmax')
parser.add_argument('--s',              type=float, default=32,     				help='Loss scale in AAM softmax')
parser.add_argument('--k',              type=float, default=0.8,     				help='percentage to divide the target domain data')
parser.add_argument('--momentum',       type=float, default=0.4,     				help='momentum value')
parser.add_argument('--eps',            type=float, default=0.01,     				help='epsilon for PGD attack')
parser.add_argument('--beta',           type=float, default=0.002,     				help='beta for PGD attack')
parser.add_argument('--n_class',        type=int,   default=800, required=False,    help='Number of speakers')
#800  for cn1  , 1641 for commonvoice

## Command
parser.add_argument('--eval',    	dest='eval', 	action='store_true', help='Only do wav pair evaluation')
## Initialization
warnings.simplefilter("ignore")

if __name__ == '__main__':

    torch.multiprocessing.set_start_method('spawn', force=True)
    #torch.multiprocessing.set_start_method(method='forkserver', force=True)
    #torch.multiprocessing.set_sharing_strategy('file_system')
    args = parser.parse_args()
    args = init_args(args)

    ## Only do CNCeleb dataset evaluation, the initial_model is necessary
    if args.eval == True:
        s = CHDAModel(**vars(args))
        logger.info("Model %s loaded from previous state!"%args.initial_model)
        s.load_parameters(args.initial_model, False)
        #EER, minDCF = s.cneval_network(eval_list = args.eval_list, eval_path = args.eval_path)
        if args.dataset == 'cnceleb':
            logger.info("Evaluate CNCeleb datasets...")
            EER, minDCF = s.cneval_network(eval_list = args.eval_list, eval_path = args.eval_path)
        elif args.dataset == 'commonvoice':
            logger.info("Evaluate CommonVoice datasets...")
            EER, minDCF = s.commoneval_network(eval_list = args.eval_list, eval_path = args.eval_path)
        logger.info("EER %2.2f%%, minDCF %.4f%%"%(EER, minDCF))
        quit()

    ## Define the data loader for training
    dataloader 	= data_loader(aug=True ,**vars(args))
    data_sampler   = torch.utils.data.RandomSampler(dataloader)
    dataloader     = torch.utils.data.DataLoader(dataloader, batch_size = args.batch_size, num_workers = args.n_cpu, sampler = data_sampler, drop_last = True)


    ## Search for the exist models
    modelfiles = glob.glob('%s/model/model_0*.model'%args.save_path)
    modelfiles.sort()

    ## If initial_model is exist, system will train from the initial_model
    if args.initial_model != "":
        logger.info("Model %s loaded from initial model!"%args.initial_model)
        s = CHDAModel(**vars(args))
        s.load_parameters(args.initial_model, True)
        epoch = 1

    ## Otherwise, system will try to start from the saved model&epoch
    elif len(modelfiles) >= 1:
        logger.info("Model %s loaded from previous state!"%modelfiles[-1])
        epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
        s = CHDAModel(**vars(args))
        s.load_parameters(modelfiles[-1], True)
    ## Otherwise, system will train from scratch
    else:
        epoch = 1
        s = CHDAModel(**vars(args))

    EERs = []
    score_save_path = os.path.join(args.save_path,'score.txt')
    score_file = open(score_save_path, "a+")

    ## Training
    score_file.write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    score_file.flush()
    while(1):
        ## Training for one epoch
        loss, lr, acc = s.train_network(epoch = epoch,  data_loader = dataloader)
        s.save_parameters(args.model_save_path + "/model_%04d.model"%epoch)
        ## Evaluation every [test_step] epochs
        if epoch % args.test_step == 0:
            logger.info("Evaluate datasets, Computing EER and minDCF ...")
            if args.dataset == 'cnceleb':
                EERs.append(s.cneval_network(eval_list = args.eval_list, eval_path = args.eval_path)[0])
            elif args.dataset == 'commonvoice':
                EERs.append(s.commoneval_network(eval_list = args.eval_list, eval_path = args.eval_path)[0])
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%"%(epoch, acc, EERs[-1], min(EERs)))
            score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%\n"%(epoch, lr, loss, acc, EERs[-1], min(EERs)))
            score_file.flush()

        if epoch >= args.max_epoch:
            quit()

        epoch += 1

    score_file.write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    score_file.flush()
    time.sleep(0.2)