from chainer.datasets import TransformDataset, tuple_dataset
import chainer
from chainer.cuda import to_gpu, to_cpu
from chainer.dataset.convert import concat_examples

import numpy as np

from transform import transform_img


def save_softlabels(data, generate_softlab, out_dir):
    soft_label = TransformDataset(data, generate_softlab)
    soft_labels = np.vstack(soft_label[0:len(soft_label)])
    np.save(out_dir, soft_labels)


def generate_softlabel(inputs, mean, std, model):
    x, lab = transform_img(inputs, mean, std, train=False)
    with chainer.using_config('train', False), \
            chainer.using_config('enable_backprop', False):
        soft_lab = to_cpu(model(to_gpu(x[None, :])).array[0])
    return soft_lab


def transform_with_softlabel(inputs, mean, std, train=False, **kwargs):
    x, soft_lab, lab = inputs
    x, lab = transform_img((x, lab), mean, std, train=train, **kwargs)
    return x, soft_lab, lab


def create_dataset_w_softlabel(inputs, softlabels_path):
    img_t, lab_t = concat_examples(inputs)
    soft_labels_t = np.load(softlabels_path)
    train_soft = tuple_dataset.TupleDataset(img_t, soft_labels_t, lab_t)
    return train_soft
