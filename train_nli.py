# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
sys.path.insert(1, '..')
from optim import get_optimizer
from utils import setup_xp, log_metrics, accuracy, regularization, update_metrics
from cuda import set_cuda

import sys
import os
import mlogger
import argparse
import numpy as np
import torch
import torch.nn as nn

from dfw import DFW
from dfw.losses import MultiClassHingeLoss, set_smoothing_enabled
from data import get_nli, get_batch, build_vocab
from mutils import adapt_grad_norm
from models import NLINet
from tqdm import tqdm


GLOVE_PATH = "dataset/GloVe/glove.840B.300d.txt"


parser = argparse.ArgumentParser(description='NLI training')
# paths
parser.add_argument("--nlipath", type=str, default='dataset/SNLI/', help="NLI data path (SNLI or MultiNLI)")
parser.add_argument("--outputdir", type=str, default='savedir/', help="Output directory")
parser.add_argument("--outputmodelname", type=str, default='model.pickle')
parser.add_argument("--server", type=str, default='http://atlas.robots.ox.ac.uk')
parser.add_argument("--port", type=int, default=9003)
parser.add_argument('--no-tqdm', dest='tqdm', action='store_false', help="use of tqdm progress bars")
parser.set_defaults(tqdm=True)


# training
parser.add_argument("--n_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
parser.add_argument("--nonlinear_fc", type=float, default=0, help="use nonlinearity in fc")
parser.add_argument("--opt", type=str, default="sgd", help="choice of optimizer")
parser.add_argument("--eta", type=float, default=0.1, help="initial learning rate")
parser.add_argument("--momentum", type=float, default=0, help="momentum")
parser.add_argument("--l2", type=float, default=0., help="l2-regularization")
parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")
parser.add_argument('--load-opt', default=None, help='data file with opt')

# model
parser.add_argument("--encoder_type", type=str, default='BLSTMEncoder', help="see list of encoders")
parser.add_argument("--enc_lstm_dim", type=int, default=2048, help="encoder nhid dimension")
parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
parser.add_argument("--n_classes", type=int, default=3, help="entailment/neutral/contradiction")
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")
parser.add_argument("--loss", type=str, default='svm', help="choice of loss function")
parser.add_argument("--smooth-svm", dest="smooth_svm", action='store_true', help="smoothness of SVM")
parser.set_defaults(smooth_svm=False)

# gpu
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
parser.add_argument("--seed", type=int, default=1234, help="seed")

params, _ = parser.parse_known_args()

# set gpu device
torch.cuda.set_device(params.gpu_id)

# print parameters passed, and all parameters
print('\ntogrep : {0}\n'.format(sys.argv[1:]))
print(params)


"""
SEED
"""
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.cuda.manual_seed(params.seed)

torch.zeros(1).cuda()  # for quick initialization of process on device

"""
DATA
"""
train, valid, test = get_nli(params.nlipath)
word_vec = build_vocab(train['s1'] + train['s2'] +
                       valid['s1'] + valid['s2'] +
                       test['s1'] + test['s2'], GLOVE_PATH)

for split in ['s1', 's2']:
    for data_type in ['train', 'valid', 'test']:
        eval(data_type)[split] = np.array([['<s>'] +
            [word for word in sent.split() if word in word_vec] +
            ['</s>'] for sent in eval(data_type)[split]])

params.word_emb_dim = 300


"""
MODEL
"""
# model config
config_nli_model = {
    'n_words'        :  len(word_vec)          ,
    'word_emb_dim'   :  params.word_emb_dim   ,
    'enc_lstm_dim'   :  params.enc_lstm_dim   ,
    'n_enc_layers'   :  params.n_enc_layers   ,
    'dpout_model'    :  params.dpout_model    ,
    'dpout_fc'       :  params.dpout_fc       ,
    'fc_dim'         :  params.fc_dim         ,
    'bsize'          :  params.batch_size     ,
    'n_classes'      :  params.n_classes      ,
    'pool_type'      :  params.pool_type      ,
    'nonlinear_fc'   :  params.nonlinear_fc   ,
    'encoder_type'   :  params.encoder_type   ,
    'use_cuda'       :  True                  ,

}

# model
encoder_types = ['BLSTMEncoder', 'BLSTMprojEncoder', 'BGRUlastEncoder',
                 'InnerAttentionMILAEncoder', 'InnerAttentionYANGEncoder',
                 'InnerAttentionNAACLEncoder', 'ConvNetEncoder', 'LSTMEncoder']
assert params.encoder_type in encoder_types, "encoder_type must be in " + \
                                             str(encoder_types)

nli_net = NLINet(config_nli_model)
print(nli_net)

# loss
if params.loss == 'svm':
    loss_fn = MultiClassHingeLoss()
else:
    loss_fn = nn.CrossEntropyLoss()

# cuda by default
nli_net.cuda()
loss_fn.cuda()

optimizer = get_optimizer(params, parameters=nli_net.parameters())

params.visdom = True
xp_name = '{}--{}--{}--eta-{}'.format(params.encoder_type, params.opt, params.loss, params.eta)
if params.smooth_svm:
    xp_name = xp_name + '--smooth'
params.log = True
params.outputdir = './saved/{}'.format(xp_name)
params.xp_name = './xp_results/{}'.format(xp_name)

if not os.path.exists(params.xp_name):
    os.makedirs(params.xp_name)
else:
    raise ValueError('Experiment already exists at {}'.format(params.xp_name))
xp = setup_xp(params, nli_net, optimizer)


"""
TRAIN
"""
val_acc_best = -1e10


def trainepoch(epoch):
    print('\nTRAINING : Epoch ' + str(epoch))
    nli_net.train()
    # shuffle the data
    permutation = np.random.permutation(len(train['s1']))

    s1 = train['s1'][permutation]
    s2 = train['s2'][permutation]
    target = train['label'][permutation]

    if epoch > 1 and params.opt == 'sgd':
        optimizer.param_groups[0]['lr'] *= params.decay
        optimizer.step_size = optimizer.param_groups[0]['lr']

    for metric in xp.train.metrics():
        metric.reset()

    for stidx in tqdm(range(0, len(s1), params.batch_size), disable=not params.tqdm,
                      desc='Train Epoch', leave=False):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[stidx:stidx + params.batch_size], word_vec)
        s2_batch, s2_len = get_batch(s2[stidx:stidx + params.batch_size], word_vec)
        s1_batch, s2_batch = s1_batch.cuda(), s2_batch.cuda()
        tgt_batch = torch.LongTensor(target[stidx:stidx + params.batch_size]).cuda()

        # model forward
        scores = nli_net((s1_batch, s1_len), (s2_batch, s2_len))
        with set_smoothing_enabled(params.smooth_svm):
            loss = loss_fn(scores, tgt_batch)

        # backward
        optimizer.zero_grad()
        loss.backward()
        if not isinstance(optimizer, DFW):
            adapt_grad_norm(nli_net, params.max_norm)
        # necessary information for the step-size of some optimizers -> provide closure
        optimizer.step(lambda: float(loss))

        # monitoring
        batch_size = scores.size(0)
        xp.train.acc.update(accuracy(scores, tgt_batch), weighting=batch_size)
        xp.train.loss.update(loss_fn(scores, tgt_batch), weighting=batch_size)
        xp.train.gamma.update(optimizer.gamma, weighting=batch_size)

    xp.train.eta.update(optimizer.eta)
    if params.l2:
        w_norm = torch.sqrt(sum(p.norm() ** 2 for p in nli_net.parameters()))
        xp.train.reg.update(0.5 * params.l2 * xp.train.weight_norm.value ** 2)
    else:
        xp.train.reg.update(0)
    xp.train.reg.update(0.5 * args.l2 * xp.train.weight_norm.value ** 2)
    xp.train.obj.update(xp.train.reg.value + xp.train.loss.value)
    xp.train.timer.update()

    print('results : epoch {0} ; mean accuracy train : {1}'
          .format(int(xp.epoch.value), xp.train.acc.value))
    print('\nEpoch: [{0}] (Train) \t'
          '({timer:.2f}s) \t'
          'Obj {obj:.3f}\t'
          'Loss {loss:.3f}\t'
          'Acc {acc:.2f}%\t'
          .format(int(xp.epoch.value),
                  timer=xp.train.timer.value,
                  acc=xp.train.acc.value,
                  obj=xp.train.obj.value,
                  loss=xp.train.loss.value))

    for metric in xp.train.metrics():
        metric.log(time=xp.epoch.value)


def evaluate(epoch, eval_type='valid', final_eval=False):
    nli_net.eval()
    global val_acc_best, lr

    if eval_type == 'valid':
        print('\nVALIDATION : Epoch {0}'.format(epoch))
        tag = 'val'
    else:
        tag = 'test'

    if tag == 'val':
        xp_group = xp.val
    else:
        xp_group = xp.test

    for metric in xp_group.metrics():
        metric.reset()

    s1 = valid['s1'] if eval_type == 'valid' else test['s1']
    s2 = valid['s2'] if eval_type == 'valid' else test['s2']
    target = valid['label'] if eval_type == 'valid' else test['label']

    for i in tqdm(range(0, len(s1), params.batch_size), disable=not params.tqdm,
                  desc='{} Epoch'.format(tag.title()), leave=False):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[i:i + params.batch_size], word_vec)
        s2_batch, s2_len = get_batch(s2[i:i + params.batch_size], word_vec)
        s1_batch, s2_batch = s1_batch.cuda(), s2_batch.cuda()
        tgt_batch = torch.LongTensor(target[i:i + params.batch_size]).cuda()

        # model forward
        scores = nli_net((s1_batch, s1_len), (s2_batch, s2_len))
        batch_size = scores.size(0)
        xp_group.acc.update(accuracy(scores, tgt_batch), weighting=batch_size)

    xp_group.timer.update()

    print('Epoch: [{0}] ({tag})\t'
          '({timer:.2f}s) \t'
          'Obj ----\t'
          'Loss ----\t'
          'Acc {acc:.2f}% \t'
          .format(int(xp.epoch.value),
                  tag=tag.title(),
                  timer=xp_group.timer.value,
                  acc=xp_group.acc.value))

    if tag == 'val':
        xp.max_val.update(xp.val.acc.value).log(time=xp.epoch.value)

    for metric in xp_group.metrics():
        metric.log(time=xp.epoch.value)

    eval_acc = xp_group.acc.value
    # save model
    if final_eval:
        print('finalgrep : accuracy {0} : {1}'.format(eval_type, eval_acc))
    else:
        print('togrep : results : epoch {0} ; mean accuracy {1} :\
              {2}'.format(epoch, eval_type, eval_acc))

    if eval_type == 'valid' and epoch <= params.n_epochs:
        if eval_acc > val_acc_best:
            print('saving model at epoch {0}'.format(epoch))
            if not os.path.exists(params.outputdir):
                os.makedirs(params.outputdir)
            torch.save(nli_net, os.path.join(params.outputdir,
                       params.outputmodelname))
            val_acc_best = eval_acc
        else:
            if 'sgd' in params.opt:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / params.lrshrink
                print('Shrinking lr by : {0}. New lr = {1}'
                      .format(params.lrshrink,
                              optimizer.param_groups[0]['lr']))


"""
Train model on Natural Language Inference task
"""
epoch = 1

with mlogger.stdout_to("{}/log.txt".format(params.xp_name)):
    while epoch <= params.n_epochs:
        xp.epoch.update(epoch)
        trainepoch(epoch)
        evaluate(epoch, 'valid')
        epoch += 1

    # Run best model on test set.
    del nli_net
    nli_net = torch.load(os.path.join(params.outputdir, params.outputmodelname))

    print('\nTEST : Epoch {0}'.format(epoch))
    evaluate(1e6, 'valid', True)
    evaluate(0, 'test', True)

    # Save encoder instead of full model
    torch.save(nli_net.encoder,
               os.path.join(params.outputdir, params.outputmodelname + '.encoder'))
