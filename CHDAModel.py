'''
This part is used to train the speaker model and evaluate the performances
'''

import torch, sys, os, numpy, soundfile, time, pickle, random, string, csv
import torch.nn as nn
from tqdm import tqdm
from tools import *
from loss import AAMsoftmax, Entropy, AMsoftmax
from model import ECAPA_TDNN
from itertools import cycle
from copy import deepcopy
from collections import OrderedDict
import torch.nn.functional as F
from torchmetrics.functional import kl_divergence
import heapq
from itertools import permutations
from infonce import InfoNCE

#from pytorch_metric_learning.losses import NTXentLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CHDAModel(nn.Module):
    def __init__(self, lr, lr_decay, C , n_class, m, s, k, momentum, eps, beta, test_step, batch_size,  **kwargs):
        super(CHDAModel, self).__init__()
        ## ECAPA-TDNN
        self.speaker_encoder	= ECAPA_TDNN(C = C).to(device)
        ## Classifier
        self.n_class			= n_class
        self.speaker_loss   	= AAMsoftmax(n_class = n_class, m = m, s = s).to(device)
        ## CHDA setting
        self.inner_lr 			= lr
        self.outer_lr  			= lr
        self.weight_name 		= [name for name, _ in list(self.named_parameters())[:]]
        self.weight_len 		= len(self.weight_name)
        self.k = k
        self.momentum = momentum
        self.eps = eps
        self.beta = beta

        self.optim           	= torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 2e-5)
        self.scheduler       	= torch.optim.lr_scheduler.StepLR(self.optim, step_size = test_step, gamma=lr_decay)
        print("Model para number = %.2f M"%(sum(param.numel() for param in self.speaker_encoder.parameters()) / 1024 / 1024))

    def train_network(self, epoch, data_loader):
        self.train()
        self.scheduler.step(epoch - 1)  ## Update the learning rate based on the current epcoh
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        total = len(data_loader)
        total_steps = 20 * total
        #inner_grouped_param  = [{'params': [p for p in self.speaker_loss.parameters()]}]
        #self.inner_optimizer = torch.optim.Adam(inner_grouped_param, lr = self.inner_lr, weight_decay = 2e-5)

        with tqdm(data_loader, total = total) as tepoch:
            for num, query in enumerate(tepoch, start = 1):
                tepoch.set_description(f"[Epoch {epoch}]")
                p = (num-1) / total_steps
                constant = 2. / (1. + numpy.exp(-10 * p)) - 1
                query_loss, query_pred = self.forward(num, query[0], query[1], query[2], constant)
                index += len(query[2])
                top1  += query_pred
                loss  += query_loss.detach().cpu().numpy()
                tepoch.set_postfix(loss=loss/(num), accuracy=(top1/index*len(query[2])).cpu().numpy(), device=device)
                #tepoch.set_postfix(loss=loss/(num), device=device)
        return loss/num, lr, top1/index*len(query[2])

    def normalize(self, data):
        #return (data - torch.min(data)) / (torch.max(data) - torch.min(data))
        return (data + torch.min(data)) / torch.sum(data + torch.min(data))

    def Entropy(self, input_):
        bs = input_.size(0)
        #epsilon = 1e-5
        epsilon = 0
        entropy = -input_ * torch.log(input_ + epsilon)
        entropy = torch.sum(entropy, dim=1)
        return entropy

    def forward(self, num, query_wx, query_sx, query_y, constant):
        query_x         = deepcopy(query_sx)
        kl_criterion	= nn.KLDivLoss(size_average=False)
        infonce_criterion = InfoNCE()

        """fast_encoder 	= deepcopy(self.speaker_encoder)
        fast_classifier = deepcopy(self.speaker_loss)
        fast_parameters	= deepcopy(list(self.parameters()))
        fast_encoder.to(device)
        fast_classifier.to(device)
        inner_grouped_param  = [{'params': [p for p in fast_parameters]}]
        inner_optimizer = torch.optim.Adam(inner_grouped_param, lr = self.inner_lr)
        fast_encoder.train()
        fast_classifier.train()"""

        # freeze encoder
        for param in self.speaker_encoder.parameters():
            param.requires_grad = False
        for param in self.fast_encoder.parameters():
            param.requires_grad = False

        stage1_labels	= torch.LongTensor(query_y).to(device)
        stage1_spec     = self.fast_encoder.covert_spec(query_x.to(device), aug = False)
        stage1_embb	    = self.fast_encoder.forward(stage1_spec)
        stage1_loss, stage1_y_pred, stage1_y_proba, stage1_softmax_out	= self.speaker_loss.forward(stage1_embb, stage1_labels)
        self.zero_grad()
        stage1_loss.backward()
        self.optim.step()

        for param in self.speaker_encoder.parameters():
            param.requires_grad = True
        for param in self.fast_encoder.parameters():
            param.requires_grad = True


        source_relevant, source_irrelevant, weak_irrelevant, y_t, topk, memory = [], [], [], [], [], []
        with torch.no_grad():
            labels		= torch.LongTensor(query_y).to(device)
            clean_spec  = self.fast_encoder.covert_spec(query_x.to(device), aug = False)
            clean_embb	= self.fast_encoder.forward(clean_spec)
            loss, y_pred, y_proba, softmax_out	= self.speaker_loss.forward(clean_embb, labels)

            weak_spec   = self.speaker_encoder.covert_spec(query_wx.to(device), aug = False)
            #weak_embb	= self.speaker_encoder.forward(weak_spec)
            max_prob, pred_u = torch.max(softmax_out, dim=-1)
            norm_sofmax      = self.Entropy(self.normalize(softmax_out))
            memory = deepcopy(norm_sofmax.detach())
            k  = round(len(memory) * self.k)
            #k  = round(len(memory) * random.uniform(0.3,0.8))
            topk = numpy.argpartition(memory.cpu(), -k)[-k:] # find top k entropy

            for i in range(len(query_x)):
                if i in topk:
                    weak_irrelevant.append(weak_spec[i])
                    y_t.append(labels[i])
                    source_irrelevant.append(query_x[i])
                else:
                    source_relevant.append(query_x[i])
            source_relevant = torch.stack(source_relevant, dim=0)
            source_irrelevant = torch.stack(source_irrelevant, dim=0)
            weak_irrelevant = torch.stack(weak_irrelevant, dim=0)
            y_t = torch.stack(y_t, dim=0)


        ## PGD Attack
        ori_x = self.speaker_encoder.covert_spec(source_irrelevant.to(device), aug = False)
        adv_x = self.speaker_encoder.covert_spec(source_irrelevant.to(device), aug = False)
        for i in range(5) :
            adv_x.requires_grad = True
            embb_adv = self.speaker_encoder.forward(adv_x)
            loss, y_pred, y_proba ,softmax_out	= self.speaker_loss.forward(embb_adv, y_t, adaption = False)
            adv_x.retain_grad()
            self.zero_grad()
            loss.backward()
            adv = adv_x + self.beta * adv_x.grad.sign()
            eta = torch.clamp(adv - ori_x, min=-self.eps, max=self.eps)
            adv_x = (ori_x + eta).detach()
        self.zero_grad()

        for name, param in self.named_parameters():
            if name == 'speaker_loss.weight':
                param.requires_grad = False

        adv_embb           = self.speaker_encoder.forward(adv_x.to(device))
        ori_embb           = self.speaker_encoder.forward(ori_x.to(device))
        weak_embb	       = self.speaker_encoder.forward(weak_irrelevant)
        #target_embedding   = torch.vstack((weak_embb, ori_embb))
        #target_labels 	   = torch.cat((y_t,y_t))

        query_spec         = self.speaker_encoder.covert_spec(query_x.to(device), aug = False)
        query_embedding    = self.speaker_encoder.forward(query_spec)
        query_labels	   = torch.LongTensor(query_y).to(device)
        target_embedding   = torch.vstack((weak_embb, query_embedding))
        target_labels 	   = torch.cat((y_t,query_labels))

        target_loss, target_y_pred, target_y_proba, target_softmax_out	= self.speaker_loss.forward(target_embedding, target_labels)
        #print('target_loss', target_loss)

        current_spec       = self.speaker_encoder.covert_spec(source_irrelevant.to(device), aug = False)
        current_embedding  = self.speaker_encoder.forward(current_spec)
        embedding = torch.vstack((current_embedding, weak_embb))
        delay_spec  = self.fast_encoder.covert_spec(source_relevant.to(device), aug = False)
        delay_embb	= self.fast_encoder.forward(delay_spec)
        delay_irrelevant_spec = self.fast_encoder.covert_spec(source_irrelevant.to(device), aug = False)
        delay_irrelevant_embb = self.fast_encoder.forward(delay_irrelevant_spec)

        delay   = torch.mean(F.normalize(delay_embb, p=2, dim=1),dim=0)
        current = torch.mean(F.normalize(embedding, p=2, dim=1),dim=0)
        kl_loss = F.kl_div(F.log_softmax(current, -1), F.softmax(delay, -1))

        weak_loss = infonce_criterion(ori_embb, weak_embb)

        #mse_loss =  torch.mean(F.cosine_similarity(F.normalize(ori_embb, p=2, dim=1),F.normalize(delay_irrelevant_embb, p=2, dim=1)))

        strong_loss = infonce_criterion(ori_embb, adv_embb)
        #print('strong_loss', strong_loss)

        delay_loss = infonce_criterion(ori_embb, delay_irrelevant_embb)
        #print('delay_loss', delay_loss)

        relevant_loss = infonce_criterion(delay_embb, delay_embb)
        #print(relevant_loss)

        speaker_loss =  target_loss  + kl_loss +  weak_loss + strong_loss + delay_loss + relevant_loss
        self.zero_grad()
        speaker_loss.backward()
        self.optim.step()
        self.momentum_step(m=self.momentum)

        for name, param in self.named_parameters():
            if name == 'speaker_loss.weight':
                param.requires_grad = True

        return target_loss, target_y_pred

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def momentum_step(self, m=1):
        params_target = self.speaker_encoder.state_dict()
        params_momentum = self.fast_encoder.state_dict()

        dict_params_momentum = dict(params_momentum)

        for name in params_target:
            theta_momentum = dict_params_momentum[name]
            theta_target = params_target[name].data
            dict_params_momentum[name].data.copy_(m * theta_momentum + (1-m) * theta_target)

        self.fast_encoder.load_state_dict(dict_params_momentum)

    def load_parameters(self, path, train):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model."%origname)
                    continue
            if train and self_state[name].size() != loaded_state[origname].size():
                ## Classifier, the number of speakers may different and must be changed according to the fine-tuning dataset
                print("clear classifier weight : dimension from %s to %s"%(loaded_state[origname].size(), self_state[name].size()))
                loaded_state[origname] = torch.nn.Parameter(torch.FloatTensor(self.n_class, 192), requires_grad=True)
                #print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            if not train and name == 'speaker_loss.weight' :
                self_state[name] = loaded_state[origname]
            self_state[name].copy_(param)
        self.fast_encoder = deepcopy(self.speaker_encoder)


    def cneval_network(self, eval_list, eval_path):
        self.eval()
        files = []
        embeddings = {}
        lines = open(eval_list).read().splitlines()
        for line in lines:
            files.append(os.path.join('enroll', '%s.flac'%line.split()[0]))
            files.append(line.split()[1].replace("wav", "flac"))
        setfiles = list(set(files))
        setfiles.sort()
        print('device : %s'%device)
        for idx, file in tqdm(enumerate(setfiles), total = len(setfiles), desc='Computing Embeddings'):
            audio, _  = soundfile.read(os.path.join(eval_path, file))
            # Full utterance
            data_1 = torch.FloatTensor(numpy.stack([audio],axis=0)).to(device)

            # Spliited utterance matrix
            max_audio = 300 * 160 + 240
            if audio.shape[0] <= max_audio:
                shortage = max_audio - audio.shape[0]
                audio = numpy.pad(audio, (0, shortage), 'wrap')
            feats = []
            startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
            for asf in startframe:
                feats.append(audio[int(asf):int(asf)+max_audio])
            feats = numpy.stack(feats, axis = 0).astype(numpy.float)
            data_2 = torch.FloatTensor(feats).to(device)
            # Speaker embeddings
            with torch.no_grad():
                spec_1      = self.speaker_encoder.covert_spec(data_1, aug = False)
                embedding_1 = self.speaker_encoder.forward(spec_1)
                embedding_1 = F.normalize(embedding_1, p=2, dim=1)
                spec_2      = self.speaker_encoder.covert_spec(data_2, aug = False)
                embedding_2 = self.speaker_encoder.forward(spec_2)
                embedding_2 = F.normalize(embedding_2, p=2, dim=1)
            embeddings[file] = [embedding_1, embedding_2]
        scores, labels  = [], []

        for line in tqdm(lines, desc='Computing EER and minDCF'):
            embedding_11, embedding_12 = embeddings[os.path.join('enroll', '%s.flac'%line.split()[0])]
            embedding_21, embedding_22 = embeddings[line.split()[1].replace("wav", "flac")]
            # Compute the scores
            score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) # higher is positive
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
            score = (score_1 + score_2) / 2
            score = score.detach().cpu().numpy()
            scores.append(score)
            labels.append(int(line.split()[2]))

        # Coumpute EER and minDCF
        EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.02, 1, 1)

        return EER, minDCF

    def commoneval_network(self, eval_list, eval_path):
        self.eval()
        files = []
        embeddings = {}
        lines = open(eval_list).read().splitlines()
        for line in lines:
            files.append(line.split()[0])
            files.append(line.split()[1])
        setfiles = list(set(files))
        setfiles.sort()
        print('device : %s'%device)
        for idx, file in tqdm(enumerate(setfiles), total = len(setfiles), desc='Computing Embeddings'):
            audio, _  = soundfile.read(os.path.join(eval_path, file))
            # Full utterance
            data_1 = torch.FloatTensor(numpy.stack([audio],axis=0)).to(device)
            # Spliited utterance matrix
            max_audio = 300 * 160 + 240
            if audio.shape[0] <= max_audio:
                shortage = max_audio - audio.shape[0]
                audio = numpy.pad(audio, (0, shortage), 'wrap')
            feats = []
            startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
            for asf in startframe:
                feats.append(audio[int(asf):int(asf)+max_audio])
            feats = numpy.stack(feats, axis = 0).astype(numpy.float)
            data_2 = torch.FloatTensor(feats).to(device)
            # Speaker embeddings
            with torch.no_grad():
                spec_1      = self.speaker_encoder.covert_spec(data_1, aug = False)
                embedding_1 = self.speaker_encoder.forward(spec_1)
                embedding_1 = F.normalize(embedding_1, p=2, dim=1)
                spec_2      = self.speaker_encoder.covert_spec(data_2, aug = False)
                embedding_2 = self.speaker_encoder.forward(spec_2)
                embedding_2 = F.normalize(embedding_2, p=2, dim=1)
            embeddings[file] = [embedding_1, embedding_2]

        scores, labels  = [], []
        for line in tqdm(lines, desc='Computing EER and minDCF'):
            embedding_11, embedding_12 = embeddings[line.split()[0]]
            embedding_21, embedding_22 = embeddings[line.split()[1]]
            # Compute the scores
            score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) # higher is positive
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
            score = (score_1 + score_2) / 2
            score = score.detach().cpu().numpy()
            scores.append(score)
            labels.append(int(line.split()[2]))

        # Coumpute EER and minDCF
        EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
        return EER, minDCF

    def visualize(self, file):
        self.eval()
        audio, _  = soundfile.read(file)
        # Full utterance
        data_1 = torch.FloatTensor(numpy.stack([audio],axis=0)).to(device)
        # Spliited utterance matrix
        max_audio = 300 * 160 + 240
        if audio.shape[0] <= max_audio:
            shortage = max_audio - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')
        feats = []
        startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])
        feats = numpy.stack(feats, axis = 0).astype(numpy.float)
        data_2 = torch.FloatTensor(feats).to(device)
        # Speaker embeddings
        with torch.no_grad():
            spec_1      = self.speaker_encoder.covert_spec(data_1, aug = False)
            embedding_1 = self.speaker_encoder.forward(spec_1)
            embedding_1 = F.normalize(embedding_1, p=2, dim=1)
            spec_2      = self.speaker_encoder.covert_spec(data_2, aug = False)
            embedding_2 = self.speaker_encoder.forward(spec_2)
            embedding_2 = F.normalize(embedding_2, p=2, dim=1)
        #embeddings[file] = [embedding_1, embedding_2]
        #vis_embeddings.append(embedding_1)

        return embedding_1