'''
This part is used to train the speaker model and evaluate the performances
'''

import torch, sys, os, tqdm, numpy, soundfile, time, pickle
import torch.nn as nn
from tools import *
from loss import AAMsoftmax, LossFunction
from model import *
import torchaudio.transforms as T
from torch.utils.data import Subset, Dataset, DataLoader
import itertools

class ECAPAModel(nn.Module):
    def __init__(self, lr, lr_decay, C , n_class, m, s, test_step, sampling_rate, model, loss, **kwargs):
        super(ECAPAModel, self).__init__()

        self.sampling_rate =  sampling_rate    
        if model == 'ecapa_tdnn':
            self.speaker_encoder = ECAPA_TDNN(C = 512, sampling_rate = self.sampling_rate).cuda()
            if loss == 'infonce':
                self.speaker_loss    = LossFunction(192,5994,'infonce').cuda() 
            elif loss == 'aam_infonce':
                self.speaker_loss    = LossFunction(192,5994,'aam_infonce').cuda() 
            elif mode == 'classifier':
                self.speaker_loss = AAMsoftmax(5994,0.2,30, 192).cuda()
        elif model == 'resnet':
            self.speaker_encoder = ResNet34(feat_dim=80, embed_dim=256, pooling_func='MQMHASTP').cuda()
            if loss == 'infonce':
                self.speaker_loss    = LossFunction(256,5994,'infonce').cuda() 
            elif loss == 'aam_infonce':
                self.speaker_loss    = LossFunction(256,5994,'aam_infonce').cuda() 
            elif mode == 'classifier':
                self.speaker_loss = AAMsoftmax(5994,0.2,30, 256).cuda()
        elif model == 'tdnn':
            self.speaker_encoder = xvecTDNN(256, 0.2).cuda()
            if loss == 'infonce':
                self.speaker_loss    = LossFunction(256,5994,'infonce').cuda() 
            elif loss == 'aam_infonce':
                self.speaker_loss    = LossFunction(256,5994,'aam_infonce').cuda() 
            elif mode == 'classifier':
                self.speaker_loss = AAMsoftmax(5994,0.2,30, 256).cuda()
        print("Margin :", m)        
        self.optim           = torch.optim.AdamW(self.speaker_encoder.parameters(), lr = lr, weight_decay = 2e-5)
        self.optim_loss           = torch.optim.AdamW(self.speaker_loss.parameters(), lr = lr, weight_decay = 2e-5)
        self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = test_step, gamma=lr_decay)
        self.scheduler2       = torch.optim.lr_scheduler.StepLR(self.optim_loss, step_size = test_step, gamma=lr_decay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.speaker_encoder.parameters()) / 1024 / 1024))
        
    def train_network(self, epoch, loader):
        self.train()
        self.scheduler.step(epoch - 1)
        self.scheduler2.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        for num, (data, labels) in enumerate(loader, start = 1):
            self.optim.zero_grad()
            self.optim_loss.zero_grad()
            labels            = torch.LongTensor(labels).cuda()
            speaker_embedding = self.speaker_encoder.forward(data.cuda(), aug = True)
            nloss, prec       = self.speaker_loss(speaker_embedding, labels)
            total_loss = nloss
            total_loss.backward()
            self.optim.step()
            self.optim_loss.step()
            index += len(labels)
            top1 += prec
            loss += nloss.detach().cpu().numpy()
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
            " Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), top1/index*len(labels)))
            sys.stderr.flush()
        sys.stdout.write("\n")
        return loss/num, lr, top1/index*len(labels)
    
    def train_clip(self, epoch, loader):
        self.train()
        self.scheduler.step(epoch - 1)
        self.scheduler2.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        idx = 0
        count = 0
        lr = self.optim.param_groups[0]['lr']
        for num, (data, data2, labels) in enumerate(loader, start = 1):
            self.optim.zero_grad()
            self.optim_loss.zero_grad()
            labels            = torch.LongTensor(labels).cuda()
            idx = idx+1
            #print(len(set(labels)))
            if len(set(labels.cpu().detach().numpy()))!=data.shape[0]:
                count = count+1
                continue
            spk_emb = self.speaker_encoder.forward(data.cuda(), aug = True)
            txt_emb = self.speaker_encoder.forward(data2.cuda(), aug = True)
            
            nloss, temp, prec       = self.speaker_loss.forward(spk_emb, txt_emb, labels)
            
            total_loss = nloss
 
            total_loss.backward()
            
            self.optim.step()
            self.optim_loss.step()
            
            index += len(labels)
            top1 += prec
            loss += nloss.detach().cpu().numpy()
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
            " Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), top1/index*len(labels)))
            sys.stderr.flush()
        sys.stdout.write("\n")
        return loss/num, lr, top1/index*len(labels)
    
    def train_blip(self, epoch, loader):
        self.train()
        self.scheduler.step(epoch - 1)
        self.scheduler2.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        idx = 0
        count = 0
        lr = self.optim.param_groups[0]['lr']
        for num, (data, data2, labels) in enumerate(loader, start = 1):
            self.optim.zero_grad()
            self.optim_loss.zero_grad()
            labels            = torch.LongTensor(labels).cuda()
            idx = idx+1
            #print(len(set(labels)))
            if len(set(labels.cpu().detach().numpy()))!=data.shape[0]:
                count = count+1
                continue
            spk_emb = self.speaker_encoder.forward(data.cuda(), aug = True)
            txt_emb = self.speaker_encoder.forward(data2.cuda(), aug = True)
            
            nloss, temp, prec       = self.speaker_loss.forward(spk_emb, txt_emb, labels)
            
           
            spk_emb = F.normalize(spk_emb, p=2, dim=1)
            txt_emb = F.normalize(txt_emb, p=2, dim=1)

            sim_i2t = spk_emb @ txt_emb.T * temp
            sim_t2i = txt_emb @ spk_emb.T * temp

            bs = data.shape[0]

            with torch.no_grad():       
                weights_t2i = F.softmax(sim_t2i[:,:bs],dim=1)+1e-4 
                weights_t2i.fill_diagonal_(0)            
                weights_i2t = F.softmax(sim_i2t[:,:bs],dim=1)+1e-4  
                weights_i2t.fill_diagonal_(0) 

            image_embeds_neg = []  
            text_ids_neg = []
            
            for b in range(bs):
                _, neg_idx = torch.max(weights_i2t[b].unsqueeze(0), 1)
                text_ids_neg.append(txt_emb[neg_idx].squeeze()) 

            for b in range(bs):
                _, neg_idx = torch.max(weights_t2i[b].unsqueeze(0), 1)
                text_ids_neg.append(txt_emb[neg_idx].squeeze()) 
            text_ids_neg = torch.stack(text_ids_neg,dim=0)

            image_embeds_all = torch.cat((spk_emb, spk_emb, spk_emb),dim=0)
            text_ids_all = torch.cat((txt_emb, text_ids_neg),dim=0)

            negative = self.speaker_encoder.itm(image_embeds_all, text_ids_all, False)

            itm_labels_pos = torch.ones(bs,dtype=torch.long).cuda()
            itm_labels_neg = torch.zeros(2*bs,dtype=torch.long).cuda()

            itm_labels = torch.cat((itm_labels_pos,itm_labels_neg),dim=0)

            loss_itm = F.cross_entropy(negative, itm_labels).cuda()

            total_loss = (0.2*loss_itm  + 0.8 * nloss) / 2
 
            total_loss.backward()
            
            self.optim.step()
            self.optim_loss.step()
            
            index += len(labels)
            top1 += prec
            loss += nloss.detach().cpu().numpy()
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
            " Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), top1/index*len(labels)))
            sys.stderr.flush()
        sys.stdout.write("\n")
        return loss/num, lr, top1/index*len(labels)

    def eval_network(self, eval_list, eval_path):
        self.eval()
        files = []
        embeddings = {}
        lines = open(eval_list).read().splitlines()
        for line in lines:
            files.append(line.split()[1])
            files.append(line.split()[2])
        setfiles = list(set(files))
        setfiles.sort()

        for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
            audio, sr  = soundfile.read(os.path.join(eval_path, file))
            audio = torch.FloatTensor(audio)
            resampler = T.Resample(sr, self.sampling_rate, dtype=audio.dtype)
            audio = resampler(audio)

            data_1 = torch.FloatTensor(numpy.stack([audio],axis=0)).cuda()

            max_audio = 300 * 160 + 240
            if audio.shape[0] <= max_audio:
                shortage = max_audio - audio.shape[0]
                audio = numpy.pad(audio, (0, shortage), 'wrap')
            feats = []
            startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
            for asf in startframe:
                feats.append(audio[int(asf):int(asf)+max_audio])
            feats = numpy.stack(feats, axis = 0).astype(numpy.float)
            data_2 = torch.FloatTensor(feats).cuda()
            with torch.no_grad():
                embedding_1 = self.speaker_encoder.forward(data_1, aug = False)
                embedding_1 = F.normalize(embedding_1, p=2, dim=1)
                embedding_2 = self.speaker_encoder.forward(data_2, aug = False)
                embedding_2 = F.normalize(embedding_2, p=2, dim=1)
            embeddings[file] = [embedding_1, embedding_2]
        scores, labels  = [], []

        for line in lines:			
            embedding_11, embedding_12 = embeddings[line.split()[1]]
            embedding_21, embedding_22 = embeddings[line.split()[2]]
            score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) # higher is positive
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
            score = (score_1 + score_2) / 2
            score = score.detach().cpu().numpy()
            scores.append(score)
            labels.append(int(line.split()[0]))

        EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

        return EER, minDCF

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model."%origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)
            
    def evaluateFromList_with_snorm(self, eval_list, eval_path, train_list, train_path, sampling_rate, top_coh_size = 20000, **kwargs):
        
        self.speaker_encoder.eval()
        self.sampling_rate = sampling_rate
        feats_eval = {}
        tstart = time.time()
        with open(eval_list) as f:
            lines_eval = f.readlines()
        files = list(itertools.chain(*[x.strip().split()[1:] for x in lines_eval]))
        setfiles = list(set(files))
        setfiles.sort()
        test_dataset = test_dataset_loader(setfiles, eval_path, sampling_rate = self.sampling_rate, **kwargs)

        sampler = None
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=10, drop_last=False, sampler=sampler)
        ds = test_loader.__len__()
        gs = 1
        for idx, data in enumerate(test_loader):
            inp1 = data[0][0].cuda()
            with torch.no_grad():
                ref_feat = self.speaker_encoder(inp1, False).detach().cpu()
            feats_eval[data[1][0]] = ref_feat
            telapsed = time.time() - tstart

            sys.stdout.write("\r Reading {:d} of {:d}: {:.2f} Hz, embedding size {:d}".format(idx*gs, ds*gs, idx*gs/telapsed,ref_feat.size()[1]))
            sys.stdout.flush()

        
        feats_coh = {}
        tstart = time.time()
        with open(train_list) as f:
            lines_coh = f.readlines()
        setfiles = list(set([x.split()[0] for x in lines_coh]))
        setfiles.sort()
        cohort_dataset = test_dataset_loader(setfiles, train_path, sampling_rate = self.sampling_rate, **kwargs)

        cohort_loader = torch.utils.data.DataLoader(cohort_dataset, batch_size=1, shuffle=False, num_workers=10, drop_last=False, sampler=sampler)
        ds = cohort_loader.__len__()
        for idx, data in enumerate(cohort_loader):
            inp1 = data[0][0].cuda()
            with torch.no_grad():
                ref_feat = self.speaker_encoder(inp1, False).detach().cpu()
            feats_coh[data[1][0]] = ref_feat
            telapsed = time.time() - tstart
           
            sys.stdout.write("\r Reading {:d} of {:d}: {:.2f} Hz, embedding size {:d}".format(idx*gs, ds*gs, idx*gs/telapsed,ref_feat.size()[1]))
            sys.stdout.flush()
        coh_feat = torch.stack(list(feats_coh.values())).squeeze(1).cuda()

        coh_feat = F.normalize(coh_feat, p=2, dim=1)

        all_scores, all_labels = [], []

        for idx, line in enumerate(lines_eval):
            data = line.split()
 
            enr_feat = feats_eval[data[1]].cuda()
            tst_feat = feats_eval[data[2]].cuda()

            enr_feat = F.normalize(enr_feat, p=2, dim=1)
            tst_feat = F.normalize(tst_feat, p=2, dim=1)

            score_e_c = F.cosine_similarity(enr_feat, coh_feat)
            score_c_t = F.cosine_similarity(coh_feat, tst_feat)
            
            score = F.cosine_similarity(enr_feat, tst_feat)

            
            top_coh_size = len(coh_feat)
            score_e_c = torch.topk(score_e_c, k=top_coh_size, dim=0)[0]
            score_c_t = torch.topk(score_c_t, k=top_coh_size, dim=0)[0]
            score_e = (score - torch.mean(score_e_c, dim=0)) / torch.std(score_e_c, dim=0)
            score_t = (score - torch.mean(score_c_t, dim=0)) / torch.std(score_c_t, dim=0)
            score = 0.5 * (score_e + score_t)

            all_scores.append(score.detach().cpu().numpy())
            all_labels.append(int(data[0]))
            telapsed = time.time() - tstart
            sys.stdout.write("\r Computing {:d} of {:d}: {:.2f} Hz".format(idx, len(lines_eval), idx/telapsed))
            sys.stdout.flush()
                    
        EER = tuneThresholdfromScore(all_scores, all_labels, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(all_scores, all_labels)
        minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
        print(EER, minDCF)
        return EER, minDCF

class test_dataset_loader(Dataset):
    
    def __init__(self, eval_list, eval_path, sampling_rate,label=False, **kwargs):
        self.test_path = eval_path
        self.test_list = eval_list
        self.test_label = label
        self.sampling_rate = sampling_rate
        
    def __getitem__(self, index):
        audio, sr  = soundfile.read(os.path.join(self.test_path, self.test_list[index]))
        audio = torch.FloatTensor(audio)
        resampler = T.Resample(sr, self.sampling_rate, dtype=audio.dtype)
        audio = resampler(audio).unsqueeze(0)
        if self.test_label!=False: 
            return torch.FloatTensor(audio), self.test_list[index], self.test_label[index]
        else:
            return torch.FloatTensor(audio), self.test_list[index]

    def __len__(self):
        return len(self.test_list)

