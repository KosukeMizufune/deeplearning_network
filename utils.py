from functools import partial
from importlib import import_module
from pathlib import Path
import os

from chainer import iterators
from chainer.datasets import cifar, split_dataset_random, TransformDataset
from chainer.links.caffe.caffe_function import CaffeFunction
from chainer.serializers import npz
import numpy as np

from transform import transform_img


def create_iterator(args):
    train_val, test = cifar.get_cifar10(scale=255.)
    train_size = int(len(train_val) * 0.9)
    train, valid = split_dataset_random(train_val, train_size, seed=0)

    mean = np.mean([x for x, _ in train], axis=(0, 2, 3))
    std = np.std([x for x, _ in train], axis=(0, 2, 3))

    transform_train = partial(transform_img, args=args, mean=mean, std=std, train=True)
    transform_valid = partial(transform_img, args=args, mean=mean, std=std)

    processed_train = TransformDataset(train, transform_train)
    processed_valid = TransformDataset(valid, transform_valid)

    train_iter = iterators.SerialIterator(processed_train, args.batchsize)
    valid_iter = iterators.SerialIterator(processed_valid,
                                          args.batchsize,
                                          repeat=False,
                                          shuffle=False)
    return train_iter, valid_iter


def create_model(model_file, model_name, npz_filename, n_class, layers):
    ext = os.path.splitext(model_file)[1]
    mod_path = '.'.join(os.path.split(model_file)).replace(ext, '')
    mod = import_module(mod_path)
    model = getattr(mod, model_name)(n_class, npz_filename, layers)
    return model


def caffe2npz(caffe_path):
    caffe_model = CaffeFunction(caffe_path)
    npz_filename = Path(caffe_path).stem + '.npz'
    npz.save_npz(npz_filename, caffe_model, compression=False)
    return npz_filename
