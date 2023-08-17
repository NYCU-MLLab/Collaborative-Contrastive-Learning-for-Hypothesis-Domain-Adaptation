'''
AAMsoftmax loss function copied from voxceleb_trainer: https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
'''

import torch, math
import torch.nn as nn
import torch.nn.functional as F
from tools import *

from copy import deepcopy

def Entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy

class GradReverse(torch.autograd.Function):
    #def __init__(self, lambd):
    #    self.lambd = lambd
    lambd = 1
    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        #return (grad_output * - GradReverse.lambd)
        return grad_output.neg()

def grad_reverse(x, lambd=1.0):
    GradReverse.lambd = lambd
    return GradReverse.apply(x)

class AAMsoftmax(nn.Module):
    def __init__(self, n_class, m, s):

        super(AAMsoftmax, self).__init__()
        self.m = m
        self.s = s
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, 192), requires_grad=True)
        self.ce = nn.CrossEntropyLoss(reduction='mean')
        nn.init.xavier_normal_(self.weight, gain=1)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label=None,  constant = 1, adaption = False):

        if adaption == True:
            x = grad_reverse(x, constant)

        cosine = F.linear(F.normalize(x), F.normalize(self.weight))

        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s


        loss = self.ce(output, label)
        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        maxk = max((1,))
        _, proba = output.detach().topk(maxk, 1, True, True)
        proba = proba.t()
        #return loss, prec1, proba , output
        return loss, prec1, proba , cosine

class AMsoftmax(nn.Module):

    def __init__(self, n_class, m, s):
        '''
        AM Softmax Loss
        '''
        super(AMsoftmax, self).__init__()
        self.s = s
        self.m = m
        self.n_class = n_class
        self.fc = nn.Linear(192, n_class, bias=False)

    def forward(self, x, label=None,  constant = 1, adaption = False):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(label)
        assert torch.min(label) >= 0
        assert torch.max(label) < self.n_class
        for W in self.fc.parameters():
            W = F.normalize(W, dim=1)

        x = F.normalize(x, dim=1)

        wf = self.fc(x)
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[label]) - self.m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(label)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        #print(torch.log(denominator))
        return -torch.mean(L)