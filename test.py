from __future__ import division
import onmt
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time
import opt

dataset = torch.load(opt.data)

trainData = onmt.Dataset(dataset['train']['src'],
                             dataset['train']['tgt'], opt.batch_size, opt.gpus)
validData = onmt.Dataset(dataset['valid']['src'],
                             dataset['valid']['tgt'], opt.batch_size, opt.gpus,
                             volatile=True)
dicts = dataset['dicts']
print(' * vocabulary size. source = %d; target = %d' %
          (dicts['src'].size(), dicts['tgt'].size()))
print(' * number of training sentences. %d' %
          len(dataset['train']['src']))
print(' * maximum batch size. %d' % opt.batch_size)

encoder = onmt.Models.Encoder(opt,dicts['src'])
