from __future__ import division
import onmt
import onmt.modules
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time
import opt

def NMTCriterion(vocabSize):
    weight = torch.ones(vocabSize)
    weight[onmt.Constants.PAD] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    if opt.gpus:
        crit.cuda()
    return crit

def memoryEfficientLoss(outputs, targets, generator, crit, eval=False):
    # compute generations one piece at a time
    num_correct, loss = 0, 0
    outputs = Variable(outputs.data, requires_grad=(not eval), volatile=eval)
    batch_size = outputs.size(1)
    outputs_split = torch.split(outputs, opt.max_generator_batches)
    targets_split = torch.split(targets, opt.max_generator_batches)
    for i, (out_t, targ_t) in enumerate(zip(outputs_split, targets_split)):
        out_t = out_t.view(-1, out_t.size(2))
        scores_t = generator(out_t)
        loss_t = crit(scores_t, targ_t.view(-1))
        pred_t = scores_t.max(1)[1]
        num_correct_t = pred_t.data.eq(targ_t.data).masked_select(targ_t.ne(onmt.Constants.PAD).data).sum()
        num_correct += num_correct_t
        loss += loss_t.data[0]
        if not eval:
            loss_t.div(batch_size).backward()
    grad_output = None if outputs.grad is None else outputs.grad.data
    return loss, grad_output, num_correct


def eval(model, criterion, data):
    total_loss = 0
    total_words = 0
    total_num_correct = 0
    model.eval()
    for i in range(len(data)):
        batch = data[i][:-1] # exclude original indices
        outputs = model(batch)
        targets = batch[1][1:]  # exclude <s> from targets
        loss, _, num_correct = memoryEfficientLoss(
                outputs, targets, model.generator, criterion, eval=True)
        total_loss += loss
        total_num_correct += num_correct
        total_words += targets.data.ne(onmt.Constants.PAD).sum()
    model.train()
    return total_loss / total_words, total_num_correct / total_words

def trainModel(model, trainData, validData, dataset, optim):
    print(model)
    model.train()
    # define criterion of each GPU
    criterion = NMTCriterion(dataset['dicts']['tgt'].size())
    start_time = time.time()
    def trainEpoch(epoch):
        if opt.extra_shuffle and epoch > opt.curriculum:
            trainData.shuffle()
        # shuffle mini batch order
        batchOrder = torch.randperm(len(trainData))
        total_loss, total_words, total_num_correct = 0, 0, 0
        report_loss, report_tgt_words, report_src_words, report_num_correct = 0, 0, 0, 0
        start = time.time()
        for i in range(len(trainData)):
            batchIdx = batchOrder[i] if epoch > opt.curriculum else i
            batch = trainData[batchIdx][:-1] # exclude original indices
            model.zero_grad()
            outputs = model(batch)
            targets = batch[1][1:]  # exclude <s> from targets
            loss, gradOutput, num_correct = memoryEfficientLoss(
                    outputs, targets, model.generator, criterion)
            outputs.backward(gradOutput)
            # update the parameters
            optim.step()
            num_words = targets.data.ne(onmt.Constants.PAD).sum()
            report_loss += loss
            report_num_correct += num_correct
            report_tgt_words += num_words
            report_src_words += batch[0][1].data.sum()
            total_loss += loss
            total_num_correct += num_correct
            total_words += num_words
            if i % opt.log_interval == -1 % opt.log_interval:
                print("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; %3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed" %
                      (epoch, i+1, len(trainData),
                      report_num_correct / report_tgt_words * 100,
                      math.exp(report_loss / report_tgt_words),
                      report_src_words/(time.time()-start),
                      report_tgt_words/(time.time()-start),
                      time.time()-start_time))
                report_loss = report_tgt_words = report_src_words = report_num_correct = 0
                start = time.time()
        return total_loss / total_words, total_num_correct / total_words
    
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')
        #  (1) train for one epoch on the training set
        train_loss, train_acc = trainEpoch(epoch)
        train_ppl = math.exp(min(train_loss, 100))
        print('Train perplexity: %g' % train_ppl)
        print('Train accuracy: %g' % (train_acc*100))
        #  (2) evaluate on the validation set
        valid_loss, valid_acc = eval(model, criterion, validData)
        valid_ppl = math.exp(min(valid_loss, 100))
        print('Validation perplexity: %g' % valid_ppl)
        print('Validation accuracy: %g' % (valid_acc*100))
        #  (3) update the learning rate
        optim.updateLearningRate(valid_loss, epoch)
        model_state_dict = model.module.state_dict() if len(opt.gpus) > 1 else model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items() if 'generator' not in k}
        generator_state_dict = model.generator.module.state_dict() if len(opt.gpus) > 1 else model.generator.state_dict()
        #  (4) drop a checkpoint
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'dicts': dataset['dicts'],
            'opt': opt,
            'epoch': epoch,
            'optim': optim
        }
        torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_e%d.pt' % (opt.save_model, 100*valid_acc, valid_ppl, epoch))


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
decoder = onmt.Models.Decoder(opt, dicts['tgt'])

vocab_dist_gen = nn.Sequential(
        nn.Linear(opt.rnn_size, dicts['tgt'].size()),
        nn.Softmax())
final_dist_gen = onmt.modules.DistGen()

model = onmt.Models.NMTModel(encoder, decoder)

if len(opt.gpus) >= 1:
    model.cuda()
    vocab_dist_gen.cuda()
    final_dist_gen.cuda()


model.vocab_dist_gen = vocab_dist_gen
model.final_dist_gen = final_dist_gen


if not opt.train_from_state_dict and not opt.train_from:
    for p in model.parameters():
        p.data.uniform_(-opt.param_init, opt.param_init)
        encoder.load_pretrained_vectors(opt)
        decoder.load_pretrained_vectors(opt)
        optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at
        )

optim.set_parameters(model.parameters())
nParams = sum([p.nelement() for p in model.parameters()])
print('* number of parameters: %d' % nParams)

#trainModel(model, trainData, validData, dataset, optim)
criterion = NMTCriterion(dataset['dicts']['tgt'].size())

# shuffle mini batch order
#batchOrder = torch.randperm(len(trainData))
total_loss, total_words, total_num_correct = 0, 0, 0
report_loss, report_tgt_words, report_src_words, report_num_correct = 0, 0, 0, 0
batchIdx = 1500
batch = trainData[batchIdx][:-1] # exclude original indices
print('src', batch[0][0].size())
print('tgt', batch[1].size())
#('src', (35L, 64L))
#('tgt', (48L, 64L))
model.zero_grad()
outputs, attns, p_gens = model(batch)
print('outputs', outputs.size())
#('outputs', (47L, 64L, 100L))  47 is the max length of target sequences in current batch
print('attns', attns.size())
print('p_gens', p_gens.size())
#('attns', (47L, 64L, 35L))
#('p_gens', (47L, 64L, 1L))
print(outputs.requires_grad, outputs.volatile)
#(True, False)
targets = batch[1][1:]  # exclude <s> from targets
sources = batch[0][0]
#loss, gradOutput, num_correct = memoryEfficientLoss(
#    outputs, targets, model.generator, criterion)

print(model)
num_correct, loss = 0, 0
batch_size = outputs.size(1)
outputs_split = torch.split(outputs, opt.max_generator_batches)
targets_split = torch.split(targets, opt.max_generator_batches)
attns_split = torch.split(attns, opt.max_generator_batches)
p_gens_split = torch.split(p_gens, opt.max_generator_batches)
print('ouputs_split', len(outputs_split), outputs_split[0].size(), outputs_split[1].size())
#('ouputs_split', 2, (32L, 64L, 100L), (15L, 64L, 100L))
print('targets_split', len(targets_split), targets_split[0].size(), targets_split[1].size())
#('targets_split', 2, (32L, 64L), (15L, 64L))
print('source size', sources.size())

crit = criterion
for i, (out_t, targ_t, attn_t, p_gen_t) in enumerate(zip(outputs_split, targets_split, attns_split, p_gens_split)):
    decoder_hidden = out_t.size(2)
    decoder_batch_len = out_t.size(0)
    out_t = out_t.view(-1, decoder_hidden)
    attn_t = attn_t.view(-1, attn_t.size(2))
    p_gen_t = p_gen_t.view(-1, p_gen_t.size(2))
    print(out_t.size(), attn_t.size(), p_gen_t.size())
    #1 (2048L, 100L)(2048L, 26L)(2048L, 1L)
    #2 (960L, 100L)(960L, 26L)(960L, 1L)
    scores_t = vocab_dist_gen(out_t)
    print(scores_t.size())
    #1 (2048L, 50003L)
    #2 (960L, 50003L)
    final_scores_t = final_dist_gen(scores_t, attn_t, p_gen_t, sources, decoder_batch_len)
    loss_t = crit(scores_t, targ_t.view(-1))
    pred_t = scores_t.max(1)[1]
    num_correct_t = pred_t.data.eq(targ_t.data).masked_select(targ_t.ne(onmt.Constants.PAD).data).sum()
    num_correct += num_correct_t
    loss += loss_t.data[0]
    if not eval:
        loss_t.div(batch_size).backward()

#outputs.backward(gradOutput)

