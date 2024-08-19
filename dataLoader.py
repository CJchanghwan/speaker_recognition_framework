import glob, numpy, os, random, soundfile, torch
from scipy import signal
import torchaudio.transforms as T
import torchaudio.compliance.kaldi as kaldi
import numpy as np
import random
import torchaudio.functional as AF

def round_down(num, divisor):
    return num - (num%divisor)

def loadWAV(filename, max_frames, sampling_rate,evalmode=True, num_eval=10, **kwargs):
    # Maximum audio length
    max_audio = max_frames * 160 + 240
    # Read wav file and convert to torch tensor
    audio, sample_rate = soundfile.read(filename)
    audio = torch.FloatTensor(audio)
    resampler = T.Resample(sample_rate, sampling_rate, dtype=audio.dtype)
    audio = resampler(audio)
    audio = audio.numpy()
    audiosize = audio.shape[0]
    if audiosize <= max_audio:
        shortage = max_audio - audiosize + 1 
        audio = np.pad(audio, (0, shortage), 'wrap')
        audiosize = audio.shape[0]
    if evalmode:
        startframe = np.linspace(0,audiosize-max_audio,num=num_eval)
    else:
        startframe = np.array([np.int64(random.random()*(audiosize-max_audio))])
    feats = []
    if evalmode and max_frames == 0:
        feats += [audio]
    else:
        for asf in startframe:
            feats += [audio[int(asf):int(asf)+max_audio]]
    feat = np.stack(feats,axis=0).astype(np.float)
    return feat

class train_loader(torch.utils.data.Dataset):
    def __init__(self, train_list, train_path, musan_path, rir_path, num_frames, sampling_rate, **kwargs):
        self.train_path = train_path
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.noisetypes = ['noise','speech','music']
        self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
        self.noiselist = {}
        augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'))
        for file in augment_files:
            if file.split('/')[-4] not in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)
        self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))
        self.data_list  = []
        self.data_label = []
        lines = open(train_list).read().splitlines()
        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        dictkeys = { key : ii for ii, key in enumerate(dictkeys) }
        for index, line in enumerate(lines):
            speaker_label = dictkeys[line.split()[0]]
            file_name     = os.path.join(train_path, line.split()[1])
            self.data_label.append(speaker_label)
            self.data_list.append(file_name)
        self.sr = sampling_rate
            
    def get_labels(self):           
        return self.data_label  

    def __getitem__(self, index):
        
        speaker1 = self.data_list[index]

        speaker2 = glob.glob(os.path.join(self.train_path, speaker1.split('/')[7])+'/*/*.wav')
  
        speaker2 = random.choice(speaker2)

        label1 = speaker1.split('/')[0]
        label2 = speaker2.split('/')[0]

        speaker = self.data_label[index]
            
        audio1 = loadWAV(speaker1,self.num_frames, self.sr, evalmode=False)  
        audio2 = loadWAV(speaker2,self.num_frames,self.sr,evalmode=False)
        
        augtype = random.randint(0,5)
        if augtype == 0:   
            audio1 = audio1
        elif augtype == 1: 
            audio1 = self.add_rev(audio1)
        elif augtype == 2: 
            audio1 = self.add_noise(audio1, 'speech')
        elif augtype == 3: 
            audio1 = self.add_noise(audio1, 'music')
        elif augtype == 4: 
            audio1 = self.add_noise(audio1, 'noise')
        elif augtype == 5: 
            audio1 = self.add_noise(audio1, 'speech')
            audio1 = self.add_noise(audio1, 'music')
            
        augtype2 = random.randint(0,5)
        if augtype2 == 0:   
            audio2 = audio2
        elif augtype2 == 1: 
            audio2 = self.add_rev(audio2)
        elif augtype2 == 2: 
            audio2 = self.add_noise(audio2, 'speech')
        elif augtype2 == 3: 
            audio2 = self.add_noise(audio2, 'music')
        elif augtype2 == 4: 
            audio2 = self.add_noise(audio2, 'noise')
        elif augtype2 == 5: 
            audio2 = self.add_noise(audio2, 'speech')
            audio2 = self.add_noise(audio2, 'music')
        return torch.FloatTensor(audio1.squeeze()), torch.FloatTensor(audio2.squeeze()), speaker

    def __len__(self):
        return len(self.data_list)

    def add_rev(self, audio):
        rir_file   = random.choice(self.rir_files)
        rir, sr     = soundfile.read(rir_file)
        rir = torch.FloatTensor(rir)
        resampler = T.Resample(sr, self.sampling_rate, dtype=rir.dtype)
        rir = resampler(rir).numpy()
        rir         = numpy.expand_dims(rir.astype(numpy.float),0)
        rir         = rir / numpy.sqrt(numpy.sum(rir**2))
        return signal.convolve(audio, rir, mode='full')[:,:self.num_frames * 160 + 240]

    def add_noise(self, audio, noisecat):
        clean_db    = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4) 
        numnoise    = self.numnoise[noisecat]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)
            noiseaudio = torch.FloatTensor(noiseaudio)
            resampler = T.Resample(sr, self.sampling_rate, dtype=noiseaudio.dtype)
            noiseaudio = resampler(noiseaudio)
            length = self.num_frames * 160 + 240
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
            start_frame = numpy.int64(random.random()*(noiseaudio.shape[0]-length))
            noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = numpy.stack([noiseaudio],axis=0)
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2)+1e-4) 
            noisesnr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True)
        return noise + audio
    
class train_dataset_sampler(torch.utils.data.Sampler):
    
    def __init__(self, data_source, num_utt, max_seg_per_spk, num_spk, distributed, seed, **kwargs):
        self.data_label = data_source.data_label
        self.num_utt = num_utt
        self.max_seg_per_spk = max_seg_per_spk
        self.num_spk = num_spk
        self.epoch = 0
        self.seed = seed
        self.distributed = distributed
        self.batch_size = num_utt * num_spk
        
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.data_label), generator=g).tolist()
        data_dict = {}
        # Sort into dictionary of file indices for each ID
        for index in indices:
            speaker_label = self.data_label[index]
            if not (speaker_label in data_dict):
                data_dict[speaker_label] = []
            data_dict[speaker_label] += [index]

        ## Group file indices for each class
        dictkeys = list(data_dict.keys())
        dictkeys.sort()
        lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]
        flattened_list = []
        flattened_label = []
        for findex, key in enumerate(dictkeys):
            data = data_dict[key]
            numSeg = round_down(min(len(data),self.max_seg_per_spk),self.num_utt)
            rp = lol(np.arange(numSeg),self.num_utt)
            flattened_label.extend([findex] * (len(rp)))
            for indices in rp:
                flattened_list += [[data[i] for i in indices]]

        ## Mix data in random order
        mixid = torch.randperm(len(flattened_label), generator=g).tolist()
        mixlabel = []
        mixmap = []

        ## Reduce data waste referred from https://github.com/clovaai/voxceleb_trainer/pull/136/files
        resmixid = []
        mixlabel_ins = 1

        if self.num_utt != 1:
            while len(mixid) > 0 and mixlabel_ins > 0:
                mixlabel_ins = 0
                for ii in mixid:
                    startbatch = round_down(len(mixlabel), self.num_spk)
                    if flattened_label[ii] not in mixlabel[startbatch:]:
                        mixlabel += [flattened_label[ii]]
                        mixmap += [ii]
                        mixlabel_ins += 1
                    else:
                        resmixid += [ii]
                mixid = resmixid
                resmixid = []
        else:
            for ii in mixid:
                startbatch = round_down(len(mixlabel), self.num_spk)
                mixlabel += [flattened_label[ii]]
                mixmap += [ii]
        mixed_list = [flattened_list[i] for i in mixmap]

        ## Divide data to each GPU
       
        total_size = round_down(len(mixed_list), self.num_spk)
        self.num_samples = total_size
        return iter(mixed_list[:total_size])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        
class train_classfier(object):
    def __init__(self, train_list, train_path, musan_path, rir_path, num_frames, sampling_rate, **kwargs):
        self.train_path = train_path
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.noisetypes = ['noise','speech','music']
        self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
        self.noiselist = {}
        augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'))
        for file in augment_files:
            if file.split('/')[-4] not in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)
        self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))
        self.data_list  = []
        self.data_label = []
        lines = open(train_list).read().splitlines()
        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        dictkeys = { key : ii for ii, key in enumerate(dictkeys) }
        for index, line in enumerate(lines):
            speaker_label = dictkeys[line.split()[0]]
            file_name     = os.path.join(train_path, line.split()[1])
            self.data_label.append(speaker_label)
            self.data_list.append(file_name)

    def __getitem__(self, index):
        audio, sr = soundfile.read(self.data_list[index])
        audio = torch.FloatTensor(audio)
        resampler = T.Resample(sr, self.sampling_rate, dtype=audio.dtype)
        audio = resampler(audio)
        length = self.num_frames * 160 + 240
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')
        start_frame = numpy.int64(random.random()*(audio.shape[0]-length))
        audio = audio[start_frame:start_frame + length]
        audio = numpy.stack([audio],axis=0)
        augtype = random.randint(0,9)
        
        if augtype == 1: 
            audio = self.add_rev(audio)
        elif augtype == 2: 
            audio = self.add_noise(audio, 'speech')
        elif augtype == 3: 
            audio = self.add_noise(audio, 'music')
        elif augtype == 4: 
            audio = self.add_noise(audio, 'noise')
        elif augtype == 5: 
            audio = self.add_noise(audio, 'speech')
            audio = self.add_noise(audio, 'music')
        else : 
            audio = audio
        return torch.FloatTensor(audio[0]), self.data_label[index]
    
    def add_rev(self, audio):
        rir_file   = random.choice(self.rir_files)
        rir, sr     = soundfile.read(rir_file)
        rir = torch.FloatTensor(rir)
        resampler = T.Resample(sr, self.sampling_rate, dtype=rir.dtype)
        rir = resampler(rir).numpy()
        rir         = numpy.expand_dims(rir.astype(numpy.float),0)
        rir         = rir / numpy.sqrt(numpy.sum(rir**2))
        return signal.convolve(audio, rir, mode='full')[:,:self.num_frames * 160 + 240]

    def add_noise(self, audio, noisecat):
        clean_db    = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4) 
        numnoise    = self.numnoise[noisecat]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)
            noiseaudio = torch.FloatTensor(noiseaudio)
            resampler = T.Resample(sr, self.sampling_rate, dtype=noiseaudio.dtype)
            noiseaudio = resampler(noiseaudio)
            length = self.num_frames * 160 + 240
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
            start_frame = numpy.int64(random.random()*(noiseaudio.shape[0]-length))
            noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = numpy.stack([noiseaudio],axis=0)
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2)+1e-4) 
            noisesnr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True)
        return noise + audio
    
    def __len__(self):
        return len(self.data_list)
    
class train_dataset_sampler(torch.utils.data.Sampler):
    
    def __init__(self, data_source, num_utt, max_seg_per_spk, num_spk, distributed, seed, **kwargs):
        self.data_label = data_source.data_label
        self.num_utt = num_utt
        self.max_seg_per_spk = max_seg_per_spk
        self.num_spk = num_spk
        self.epoch = 0
        self.seed = seed
        self.distributed = distributed
        self.batch_size = num_utt * num_spk
        
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.data_label), generator=g).tolist()
        data_dict = {}
        # Sort into dictionary of file indices for each ID
        for index in indices:
            speaker_label = self.data_label[index]
            if not (speaker_label in data_dict):
                data_dict[speaker_label] = []
            data_dict[speaker_label] += [index]

        ## Group file indices for each class
        dictkeys = list(data_dict.keys())
        dictkeys.sort()
        lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]
        flattened_list = []
        flattened_label = []
        for findex, key in enumerate(dictkeys):
            data = data_dict[key]
            numSeg = round_down(min(len(data),self.max_seg_per_spk),self.num_utt)
            rp = lol(np.arange(numSeg),self.num_utt)
            flattened_label.extend([findex] * (len(rp)))
            for indices in rp:
                flattened_list += [[data[i] for i in indices]]

        ## Mix data in random order
        mixid = torch.randperm(len(flattened_label), generator=g).tolist()
        mixlabel = []
        mixmap = []

        ## Reduce data waste referred from https://github.com/clovaai/voxceleb_trainer/pull/136/files
        resmixid = []
        mixlabel_ins = 1

        if self.num_utt != 1:
            while len(mixid) > 0 and mixlabel_ins > 0:
                mixlabel_ins = 0
                for ii in mixid:
                    startbatch = round_down(len(mixlabel), self.num_spk)
                    if flattened_label[ii] not in mixlabel[startbatch:]:
                        mixlabel += [flattened_label[ii]]
                        mixmap += [ii]
                        mixlabel_ins += 1
                    else:
                        resmixid += [ii]
                mixid = resmixid
                resmixid = []
        else:
            for ii in mixid:
                startbatch = round_down(len(mixlabel), self.num_spk)
                mixlabel += [flattened_label[ii]]
                mixmap += [ii]
        mixed_list = [flattened_list[i] for i in mixmap]

        ## Divide data to each GPU
       
        total_size = round_down(len(mixed_list), self.num_spk)
        self.num_samples = total_size
        return iter(mixed_list[:total_size])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        
