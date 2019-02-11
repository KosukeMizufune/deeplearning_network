from chainer.datasets import TransformDataset
import chainer
from chainer.cuda import to_gpu, to_cpu

import numpy as np

from .. import transform


def save_softlabels(data, generate_softlab, out_dir):
    soft_label = TransformDataset(data, generate_softlab)
    soft_labels = np.vstack(soft_label[0:len(soft_label)])
    np.save(out_dir, soft_labels)


def generate_softlabel(inputs, mean, std, model, train=False, **kwargs):
    x, lab = transform.transform(inputs, mean, std, train, **kwargs)
    with chainer.using_config('train', False), \
            chainer.using_config('enable_backprop', False):
        soft_lab = to_cpu(model(to_gpu(x[None, :])).array[0])
    return soft_lab
