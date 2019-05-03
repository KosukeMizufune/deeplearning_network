import os
from functools import partial
import argparse

from chainer import serializers
from chainer.datasets import cifar, split_dataset_random
import numpy as np

from utils import create_model
from distill.utils import generate_softlabel, save_softlabels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher_file', type=str, default=None)
    parser.add_argument('--teacher_name', type=str, default=None)
    parser.add_argument('--teacher_model', type=str, default=None)
    parser.add_argument('--softlabels_path', type=str, default=None)

    args = parser.parse_args()

    train_val, test = cifar.get_cifar10(scale=255.)
    train_size = int(len(train_val) * 0.9)
    train, valid = split_dataset_random(train_val, train_size, seed=0)

    mean = np.mean([x for x, _ in train], axis=(0, 2, 3))
    std = np.std([x for x, _ in train], axis=(0, 2, 3))
    n_class = 10

    teacher = create_model(args.teacher_file,
                           args.teacher_name,
                           n_class)
    serializers.load_npz(
        args.teacher_model,
        teacher, path='updater/model:main/predictor/')
    teacher.to_gpu()

    fun_generate_soft = partial(generate_softlabel, model=teacher, mean=mean, std=std)

    if not os.path.exists(args.softlabels_path):
        save_softlabels(train, fun_generate_soft, args.softlabels_path)
    if not os.path.exists('val_' + args.softlabels_path):
        save_softlabels(valid, fun_generate_soft, 'val_' + args.softlabels_path)
