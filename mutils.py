# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import torch
import mlogger


def save_state(model, optimizer, filename):
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}, filename)


def setup_xp(args, model, optimizer):

    env_name = args.xp_name.split('/')[-1]
    if args.visdom:
        plotter = mlogger.VisdomPlotter({'env': env_name, 'server': args.server, 'port': args.port})
    else:
        plotter = None

    xp = mlogger.Container()

    xp.config = mlogger.Config(plotter=plotter, **vars(args))

    xp.epoch = mlogger.metric.Simple()

    xp.train = mlogger.Container()
    xp.train.acc = mlogger.metric.Average(plotter=plotter, plot_title="Accuracy", plot_legend="training")
    xp.train.loss = mlogger.metric.Average(plotter=plotter, plot_title="Objective", plot_legend="loss")
    xp.train.obj = mlogger.metric.Simple(plotter=plotter, plot_title="Objective", plot_legend="objective")
    xp.train.reg = mlogger.metric.Simple(plotter=plotter, plot_title="Objective", plot_legend="regularization")
    xp.train.weight_norm = mlogger.metric.Simple(plotter=plotter, plot_title="Weight-Norm")
    xp.train.step_size = mlogger.metric.Average(plotter=plotter, plot_title="Step-Size", plot_legend="clipped")
    xp.train.step_size_u = mlogger.metric.Average(plotter=plotter, plot_title="Step-Size", plot_legend="unclipped")
    xp.train.timer = mlogger.metric.Timer(plotter=plotter, plot_title="Time", plot_legend='training')

    xp.val = mlogger.Container()
    xp.val.acc = mlogger.metric.Average(plotter=plotter, plot_title="Accuracy", plot_legend="validation")
    xp.val.timer = mlogger.metric.Timer(plotter=plotter, plot_title="Time", plot_legend='validation')
    xp.max_val = mlogger.metric.Maximum(plotter=plotter, plot_title="Accuracy", plot_legend='best-validation')

    xp.test = mlogger.Container()
    xp.test.acc = mlogger.metric.Average(plotter=plotter, plot_title="Accuracy", plot_legend="test")
    xp.test.timer = mlogger.metric.Timer(plotter=plotter, plot_title="Time", plot_legend='test')

    if args.visdom:
        plotter.set_win_opts("Step-Size", {'ytype': 'log'})
        plotter.set_win_opts("Objective", {'ytype': 'log'})

    if args.log:
        # log at each epoch
        xp.epoch.hook_on_update(lambda: xp.save_to('{}/results.json'.format(args.xp_name)))
        xp.epoch.hook_on_update(lambda: save_state(model, optimizer, '{}/model.pkl'.format(args.xp_name)))

        # log after final evaluation on test set
        xp.test.acc.hook_on_update(lambda: xp.save_to('{}/results.json'.format(args.xp_name)))
        xp.test.acc.hook_on_update(lambda: save_state(model, optimizer, '{}/model.pkl'.format(args.xp_name)))

        # save results and model for best validation performance
        xp.max_val.hook_on_new_max(lambda: save_state(model, optimizer, '{}/best_model.pkl'.format(args.xp_name)))

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
