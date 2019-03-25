# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import torch
import logger


def get_xp(args):
    env_name = args.xp_name.split('/')[-1]
    plotter = logger.Plotter({'env': env_name, 'server': args.server, 'port': args.port})
    xp = logger.Experiment(args.xp_name, plotter)

    xp.epoch = logger.SimpleMetric()
    xp.acc = logger.AvgMetric()
    xp.best_acc = logger.BestMetric()
    xp.timer = logger.TimeMetric()
    xp.obj = logger.AvgMetric()
    xp.step_size = logger.AvgMetric()

    xp.train_metrics = (xp.timer, xp.step_size, xp.acc, xp.obj)
    xp.eval_metrics = (xp.timer, xp.acc)

    xp.config.update(**vars(args))
    xp.config.record()

    if args.visdom:
        plotter.set_win_opts("step_size", {'ytype': 'log'})
        xp.best_acc.link_plotter(plotter, plot_title='acc')

    return xp


@torch.autograd.no_grad()
def accuracy(out, targets, topk=1):
    if topk == 1:
        _, pred = torch.max(out, 1)
        acc = torch.mean(torch.eq(pred, targets).float())
    else:
        _, pred = out.topk(topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        acc = correct[:topk].view(-1).float().sum(0) / out.size(0)

    return 100. * acc


def adapt_grad_norm(model, max_grad_norm=None):

    if max_grad_norm is None:
        return

    shrink_factor = 1
    total_norm = 0
    for p in model.parameters():
        if p.requires_grad:
            total_norm += p.grad.data.norm().item() ** 2
    total_norm = np.sqrt(total_norm)
    if total_norm > max_grad_norm:
        shrink_factor = max_grad_norm / total_norm
    for p in model.parameters():
        if p.requires_grad:
            p.grad.data *= shrink_factor

"""
Importing batcher and prepare for SentEval
"""


def batcher(batch, params):
    # batch contains list of words
    batch = [['<s>'] + s + ['</s>'] for s in batch]
    sentences = [' '.join(s) for s in batch]
    embeddings = params.infersent.encode(sentences, bsize=params.batch_size,
                                         tokenize=False)

    return embeddings


def prepare(params, samples):
    params.infersent.build_vocab([' '.join(s) for s in samples],
                                 params.glove_path, tokenize=False)


class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
