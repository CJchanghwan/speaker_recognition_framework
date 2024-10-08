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

# Pretrained model

We provide the following pre-trained `VANILA_ECAPA-TDNN` and `ECAPA-TDNN(trained with our method)`[pretrained_weight](https://drive.google.com/drive/folders/1cszCCaU2NpIZtliy92VfD0I89Zxn6cNK?usp=drive_link)

# Preparation
Please follow the official code to prepare your VoxCeleb2 dataset from the 'Data preparation' part in this repository : [voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer).

Dataset for training usage:

1. VoxCeleb2 training set;

2. MUSAN dataset;

3. RIR dataset.

Dataset for test usage:

1. VoxCeleb1-O

2. VoxCeleb1-E

3. VoxCeleb1-H


# Extract speaker embedding
If you are using our code to extract speaker embeddings, use it like this:

```python
 import model
 import torch
 import torch, sys, os, tqdm, numpy, soundfile, time, pickle
 import torchaudio.transforms as T
 import torch.nn.functional as F
 
 use_cuda = torch.cuda.is_available() and True
 device = torch.device("cuda" if use_cuda else "cpu")
 
 speaker_encoder = model.ECAPA_TDNN(1024, 16000).to(device)
 
 loaded_state = torch.load('/path/pretrained_model.model')
 self_state = speaker_encoder.state_dict()
 new_state_dict = {name.replace("speaker_encoder.", ""): param for name, param in loaded_state.items()}
 new_state_dict.pop('speaker_loss.weight')
 
 speaker_encoder.load_state_dict(new_state_dict)
 sampling_rate = 16000
 audio_path = 'your audio path'
 audio, sr  = soundfile.read(os.path.join(audio_path))
 
 audio = torch.FloatTensor(audio)
 resampler = T.Resample(sr, sampling_rate, dtype=audio.dtype)
 audio = resampler(audio)

 with torch.no_grad():
     speaker_encoder.eval()
     embedding = speaker_encoder(audio.unsqueeze(0).to(device), False) # audio = [B, sampling_rate * second]
     embedding = F.normalize(embedding, p=2, dim=1) # embedding = [B, 192]
 print(embedding.shape)
```

# Cosine similarity between speaker embeddings
This is a method of measuring cosine similarity between two utterances.

```python
sampling_rate = 16000
audio_path1 = 'audio path1'
audio_path2 = 'audio path2'

audio1, sr  = soundfile.read(os.path.join(audio_path1))
audio2, sr  = soundfile.read(os.path.join(audio_path2))

audio1 = torch.FloatTensor(audio1)
resampler = T.Resample(sr, sampling_rate, dtype=audio.dtype)
audio1 = resampler(audio1)

audio2 = torch.FloatTensor(audio2)
resampler = T.Resample(sr, sampling_rate, dtype=audio.dtype)
audio2 = resampler(audio2)

with torch.no_grad():
    speaker_encoder.eval()
    embedding_11 = speaker_encoder(audio1.unsqueeze(0).to(device), False)
    embedding_11 = F.normalize(embedding_11, p=2, dim=1)
    
    embedding_21 = speaker_encoder(audio2.unsqueeze(0).to(device), False)
    embedding_21 = F.normalize(embedding_21, p=2, dim=1)

score = F.cosine_similarity(embedding_11.to(device), embedding_21.to(device))

print(score)
```

# Training 

you must change the data path in the `trainECAPAModel.py`

```sh
!CUDA_VISIBLE_DEVICES=0 python trainECAPAModel.py \
    --sampling_rate sampling rate \ # audio resampling
    --eval_list /path/veri_test2.txt \ # validation dataset path
    --save_path ./train_log \ # model save path
    --mode blip \ # training method(classifier, clip, blip)
    --loss aam_infonce \ # loss function(aamsoftmax, infonce, aam_infonce)
    --model ecapa-tdnn \ # model(vanila-ecapa-tdnn(only classifier), ecapa-tdnn, resnet18, resnet34, resnet50, resnet101, resnet152, resnet221, resnet293, tdnn)
    --initial_model /path/model/save/initial_weight.model \ # load initial weight
```
    
This repository provides code to train three models : (ECAPA-TDNN, Resnet, Tdnn) using the classification approach and contrastive learning approach.        
The result will be saved in `./train_log/score.txt` The model will saved in `./train_log/model`

- Example of `classification approach` ECAPA-TDNN with AAM-Softmax :

```bash
!CUDA_VISIBLE_DEVICES=0 python trainECAPAModel.py \
    --sampling_rate 16000 \
    --eval_list /path/veri_test2.txt \
    --save_path ./train_log \ 
    --mode classifier \
    --loss aamsoftmax \
    --model ecapa-tdnn \
    --initial_model /path/model/save/initial_weight.model \
```

- Example of `constrastive learning approach(CLIP)` training ECAPA-TDNN with InfoNCE  :

```bash
!CUDA_VISIBLE_DEVICES=0 python trainECAPAModel.py \
    --sampling_rate 16000 \
    --eval_list /path/veri_test2.txt \
    --save_path ./train_log \ 
    --mode clip \
    --loss infonce \ 
    --model ecapa-tdnn \
    --initial_model /path/model/save/initial_weight.model \
```

- Example of `constrastive learning approach(with hard negative sampling)` training ECAPA-TDNN with AAM-InfoNCE  :

```bash
!CUDA_VISIBLE_DEVICES=0 python trainECAPAModel.py \
    --sampling_rate 16000 \
    --eval_list /path/veri_test2.txt \
    --save_path ./train_log \ 
    --mode blip \
    --loss aam_infonce \ 
    --model ecapa-tdnn \
    --initial_model /path/model/save/initial_weight.model \
```
         
# Inferencing

The following is a usage of performing performance evaluation of EER and min-dcf. In the case of `--snorm`, AS-Norm is performed.

```bash
!CUDA_VISIBLE_DEVICES=0 python trainECAPAModel.py \
    --eval \
    #--snorm \ 
    --sampling_rate 16000 \
    --eval_list /path/list_test_all2.txt \
    --model your_model \
    #--train_list /path/train_vox2.txt \
    #--train_path /path/dev/aac/ \
    --initial_model /path/model/save/saved_weight.model
```
In the case of AS-Norm, it takes a lot of time, and in our case, it took more than 24 hours on Nvidia A5000.

# License
This code is MIT-licensed. The license applies to our pre-trained models as well.
