# Contrastive Speaker Representation Learning with Hard Negative Sampling for Speaker Recognition

# Introduction

This repository provides a PyTorch implementation of speaker recognition framework
![cssl](cssl.png)
This repository is modified based on : [ECAPA-TDNN](https://github.com/taoruijie/ecapa-tdnn)
# Dependencies

- Python3
- Numpy
- PyTorch
- librosa
- tqdm
- torchaudio
- ECAPA-TDNN
- Resnet
- Tdnn
- Clip

# Data preparation
Please follow the official code to perpare your VoxCeleb2 dataset from the 'Data preparation' part in this repository : [voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer).

Dataset for training usage:

1. VoxCeleb2 training set;

2. MUSAN dataset;

3. RIR dataset.

Dataset for test usage:

1. Voxceleb1-O

2. Voxceleb1-E

3. Voxceleb1-H


# Pre-trained models

We provide the following pre-trained ECAPA-TDNN: [ECAPA-TDNN_pretrained_weight](https://drive.google.com/drive/folders/1cszCCaU2NpIZtliy92VfD0I89Zxn6cNK?usp=drive_link)

# Training 

you must change the data path in the trainECAPAModel.py


    !CUDA_VISIBLE_DEVICES=0 python trainECAPAModel.py \
        --sampling_rate 16000 \ # audio resampling
        --eval_list /workspace/data/chgo/veri_test2.txt\ # validation dataset path
        --save_path /workspace/data/chgo/exps8k/exp1\ # model save path
        --mode blip \ # training method(classifier, clip, blip)
        --loss aam_infonce \ # loss function(infonce, aam_infonce)
        --model tdnn \ # model(ECAPA-TDNN, Resnet, Tdnn)
        --initial_model /workspace/data/chgo/voxceleb_code/ECAPA-TDNN-main_8k/exps/ \# load pretrainned weight

