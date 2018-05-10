# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import torch


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
