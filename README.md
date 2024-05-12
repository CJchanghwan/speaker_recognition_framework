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

# Extract speaker embedding
   
    import model
    import torch
    import torch, sys, os, tqdm, numpy, soundfile, time, pickle
    import torchaudio.transforms as T
    import torch.nn.functional as F
    
    use_cuda = torch.cuda.is_available() and True
    device = torch.device("cuda" if use_cuda else "cpu")
    
    speaker_encoder = model.ECAPA_TDNN(512, 16000).to(device)
    
    loaded_state = torch.load('/path/ecapa-tdnn.model')
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
        embedding = speaker_encoder(audio.unsqueeze(0).to(device), False)
        embedding = F.normalize(embedding, p=2, dim=1)
    print(embedding.shape)

```python
print("Hello, World!")
```


# Training 

you must change the data path in the trainECAPAModel.py


    !CUDA_VISIBLE_DEVICES=0 python trainECAPAModel.py \
        --sampling_rate sampling rate \ # audio resampling
        --eval_list /path/veri_test2.txt \ # validation dataset path
        --save_path /path/model/save/directory \ # model save path
        --mode blip \ # training method(classifier, clip, blip)
        --loss aam_infonce \ # loss function(infonce, aam_infonce)
        --model tdnn \ # model(ECAPA-TDNN, Resnet, Tdnn)
        --initial_model /path/model/save/saved_weight.model \# load pretrainned weight

This repository provides code to train three models : (ECAPA-TDNN, Resnet, Tdnn) using the classification approach and contrastive learning approach.        
The result will be saved in /path/model/save/directory/score.txt The model will saved in /path/model/save/model

# Inferencing

The following is a usage of performing performance evaluation of EER and min-dcf. In the case of --snorm, AS-Norm is performed.

    !CUDA_VISIBLE_DEVICES=0 python trainECAPAModel.py \
        --eval \
        #--snorm \ 
        --sampling_rate 16000 \
        --eval_list /path/veri_test2.txt \
        #--train_list /path/train_vox2.txt \
        #--train_path /path/dev/aac/ \
        --initial_model /path/model/save/saved_weight.model

With AS-norm, this system performs EER: 0.86. We will not update this code recently since no enough time for this work. I suggest you the following paper if you want to add AS-norm or other norm methods:
In the case of AS-Norm, it takes a lot of time, and in our case, it took more than 24 hours on Nvidia A5000.

