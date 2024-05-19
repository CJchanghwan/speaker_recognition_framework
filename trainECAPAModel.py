import argparse, glob, os, torch, warnings, time
from tools import *
from dataLoader import *
from ECAPAModel import ECAPAModel

parser = argparse.ArgumentParser(description = "ECAPA_trainer")
## Training Settings
parser.add_argument('--num_frames', type=int,   default=200,     help='Duration of the input segments, eg: 200 for 2 second')
parser.add_argument('--max_epoch',  type=int,   default=300,      help='Maximum number of epochs')
parser.add_argument('--batch_size', type=int,   default=100,     help='Batch size')
parser.add_argument('--n_cpu',      type=int,   default=10,       help='Number of loader threads')
parser.add_argument('--test_step',  type=int,   default=1,       help='Test and save every [test_step] epochs')
parser.add_argument('--lr',         type=float, default=0.001,   help='Learning rate')
parser.add_argument("--lr_decay",   type=float, default=0.97,    help='Learning rate decay every [test_step] epochs')
parser.add_argument("--sampling_rate", type=int,   default=16000,     help='Input sampling rate')
parser.add_argument("--model", type=str,   default='ecapa_tdnn',     help='Input model name')
parser.add_argument("--mode", type=str,   default='blip',     help='training method')

parser.add_argument('--num_spk',        type=int,   default=101,    help='Number of speakers per batch, i.e., batch size = num_spk * num_utt')
parser.add_argument('--num_utt',        type=int,   default=1,      help='Number of utterances per speaker in batch')
parser.add_argument('--max_seg_per_spk',type=int,   default=100,    help='Maximum number of utterances per speaker per epoch')
parser.add_argument('--distributed',    dest='distributed', action='store_true', help='Enable distributed training')
parser.add_argument('--loss',  type=str, default='infonce', help='Input loss')

## Training and evaluation path/lists, save path
parser.add_argument('--train_list', type=str,   default="/workspace/data/chgo/voxceleb_code/ECAPA-TDNN-main/train_list.txt",     help='The path of the training list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt')
parser.add_argument('--train_path', type=str,   default="/workspace/data/chgo/voxceleb_code/dev/aac/",                    help='The path of the training data, eg:"/data08/VoxCeleb2/train/wav" in my case')
parser.add_argument('--eval_list',  type=str,   default="/workspace/data/chgo/veri_test.txt",              help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_all2.txt')
parser.add_argument('--eval_path',  type=str,   default="/workspace/data/chgo/voxceleb_code/wav/",                    help='The path of the evaluation data, eg:"/workspace/data/chgo/wav/" in my case')
parser.add_argument('--musan_path', type=str,   default="/workspace/data/chgo/voxceleb_code/musan_split",                    help='The path to the MUSAN set, eg:"/data08/Others/musan_split" in my case')
parser.add_argument('--rir_path',   type=str,   default="/workspace/data/chgo/voxceleb_code/RIRS_NOISES/simulated_rirs",     help='The path to the RIR set, eg:"/data08/Others/RIRS_NOISES/simulated_rirs" in my case');
parser.add_argument('--save_path',  type=str,   default="exps/exp1",                                     help='/workspace/data/chgo/tdnn_new.pt')
parser.add_argument('--initial_model',  type=str,   default="",                                          help='Path of the initial_model')

## Model and Loss settings
parser.add_argument('--C',       type=int,   default=1024,   help='Channel size for the speaker encoder')
parser.add_argument('--m',       type=float, default=0.2,    help='Loss margin in AAM softmax')
parser.add_argument('--s',       type=float, default=30,     help='Loss scale in AAM softmax')
parser.add_argument('--n_class', type=int,   default=5994,   help='Number of speakers')
parser.add_argument('--seed',           type=int,   default=10,     help='Seed for the random number generator')

## Command
parser.add_argument('--eval',    dest='eval', action='store_true', help='Only do evaluation')
parser.add_argument('--snorm',    dest='snorm', action='store_true', help='Only do evaluation')

## Initialization
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()
args = init_args(args)

## Define the data loader
trainloader = train_loader(**vars(args))
train_sampler = train_dataset_sampler(trainloader, **vars(args))

from torch.utils.data import Subset, Dataset, DataLoader
from torchsampler.imbalanced import ImbalancedDatasetSampler

train_subsampler = ImbalancedDatasetSampler(trainloader)

def pad_sequence(sequences, batch_first=False, padding_value=0):

    audio_padded_sequences = []
    audio_padded2_sequences = []
    target_padded_sequences = []

    for i, seq in enumerate(sequences):
        if seq[2] not in target_padded_sequences:
            audio_padded_sequences.append(seq[0])
            audio_padded2_sequences.append(seq[1])
            target_padded_sequences.append(seq[2])
            if len(target_padded_sequences) == 200:
                break
    audio_padded_sequences= torch.stack(audio_padded_sequences,dim=0)
    audio_padded2_sequences= torch.stack(audio_padded2_sequences,dim=0)

    return audio_padded_sequences, audio_padded2_sequences, torch.LongTensor(target_padded_sequences)

trainLoader = torch.utils.data.DataLoader(trainloader, batch_size = args.batch_size, shuffle = True, num_workers = args.n_cpu, drop_last = True, collate_fn=lambda x: pad_sequence(x, batch_first=True))
## Search for the exist models
modelfiles = glob.glob('%s/model_0*.model'%args.model_save_path)
modelfiles.sort()

## Only do evaluation, the initial_model is necessary
if args.eval == True and args.snorm ==False:
	s = ECAPAModel(**vars(args))
	print("Model %s loaded from previous state!"%args.initial_model)
	s.load_parameters(args.initial_model)
	EER, minDCF = s.eval_network(eval_list = args.eval_list, eval_path = args.eval_path)
	print("EER %2.2f%%, minDCF %.4f%%"%(EER, minDCF))
	quit()
elif args.eval == True and args.snorm == True:
	s = ECAPAModel(**vars(args))
	print("Model %s loaded from previous state!"%args.initial_model)
	s.load_parameters(args.initial_model)
	EER, minDCF = s.evaluateFromList_with_snorm(eval_list = args.eval_list, eval_path = args.eval_path, train_list = args.train_list, train_path = args.train_path, sampling_rate = args.sampling_rate )
	print("EER %2.2f%%, minDCF %.4f%%"%(EER, minDCF))
	quit()

## If initial_model is exist, system will train from the initial_model
if args.initial_model != "":
	print("Model %s loaded from previous state!"%args.initial_model)
	s = ECAPAModel(**vars(args))
	s.load_parameters(args.initial_model)
	epoch = 1

## Otherwise, system will try to start from the saved model&epoch
elif len(modelfiles) >= 1:
	print("Model %s loaded from previous state!"%modelfiles[-1])
	epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
	s = ECAPAModel(**vars(args))
	s.load_parameters(modelfiles[-1])
## Otherwise, system will train from scratch
else:
	epoch = 1
	s = ECAPAModel(**vars(args))

EERs = []
score_file = open(args.score_save_path, "a+")

if args.mode =='clip':
    while(1):
        ## Training for one epoch
        loss, lr, acc = s.train_clip(epoch = epoch, loader = trainLoader)

        ## Evaluation every [test_step] epochs
        if epoch % args.test_step == 0:
            s.save_parameters(args.model_save_path + "/model_%04d.model"%epoch)
            EERs.append(s.eval_network(eval_list = args.eval_list, eval_path = args.eval_path)[0])
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%"%(epoch, acc, EERs[-1], min(EERs)))
            score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%\n"%(epoch, lr, loss, acc, EERs[-1], min(EERs)))
            score_file.flush()

        if epoch >= args.max_epoch:
            quit()

        epoch += 1

elif args.mode == 'blip':
    while(1):
        ## Training for one epoch
        loss, lr, acc = s.train_blip(epoch = epoch, loader = trainLoader)

        ## Evaluation every [test_step] epochs
        if epoch % args.test_step == 0:
            s.save_parameters(args.model_save_path + "/model_%04d.model"%epoch)
            EERs.append(s.eval_network(eval_list = args.eval_list, eval_path = args.eval_path)[0])
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%"%(epoch, acc, EERs[-1], min(EERs)))
            score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%\n"%(epoch, lr, loss, acc, EERs[-1], min(EERs)))
            score_file.flush()

        if epoch >= args.max_epoch:
            quit()

        epoch += 1
        
elif args.mode == 'classifier':
    trainloader = train_classfier(**vars(args))
    trainLoader = torch.utils.data.DataLoader(trainloader, batch_size = args.batch_size, shuffle = True, num_workers = args.n_cpu, drop_last = True)
    while(1):
        ## Training for one epoch
        loss, lr, acc = s.train_network(epoch = epoch, loader = trainLoader)

        ## Evaluation every [test_step] epochs
        if epoch % args.test_step == 0:
            s.save_parameters(args.model_save_path + "/model_%04d.model"%epoch)
            EERs.append(s.eval_network(eval_list = args.eval_list, eval_path = args.eval_path)[0])
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%"%(epoch, acc, EERs[-1], min(EERs)))
            score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%\n"%(epoch, lr, loss, acc, EERs[-1], min(EERs)))
            score_file.flush()

        if epoch >= args.max_epoch:
            quit()

        epoch += 1
