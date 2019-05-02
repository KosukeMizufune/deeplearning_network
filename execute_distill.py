import argparse
import os
from functools import partial

import chainer
from chainer.training import extensions
from chainer import links as L
from chainer import serializers
from chainer.datasets import cifar, split_dataset_random, tuple_dataset
from chainer.dataset.convert import concat_examples

import numpy as np

from utils import create_iterator, create_model, create_trainer, caffe2npz, trainer_extend
from distill.utils import generate_softlabel, save_softlabels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default='models/squeezenet.py')
    parser.add_argument('--model_name', type=str, default='SqueezeNet')
    parser.add_argument('--gpu_id', type=int, default=0)

    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--epoch_or_iter', type=str, default='epoch')
    parser.add_argument('--num_epochs_or_iter', type=int, default=500)
    parser.add_argument('--initial_lr', type=float, default=0.05)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--lr_decay_epoch', type=float, nargs='*', default=25)
    parser.add_argument('--freeze_layer', type=str, nargs='*', default=None)
    parser.add_argument('--small_lr_layers', type=str, nargs='*', default=None)
    parser.add_argument('--small_initial_lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0.0005)

    parser.add_argument('--save_dir', type=str, default='result')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--caffe_model_path', type=str, default=None)
    parser.add_argument('--save_trainer_interval', type=int, default=10)

    parser.add_argument('--layers', type=str, nargs='*', default=None)

    parser.add_argument('--pca_sigma', type=float, default=255.)
    parser.add_argument('--random_angle', type=int, default=15)
    parser.add_argument('--expand_ratio', type=float, default=1.2)
    parser.add_argument('--x_random_flip', type=bool, default=True)
    parser.add_argument('--y_random_flip', type=bool, default=False)
    parser.add_argument('--random_erase', type=bool, default=False)
    parser.add_argument('--random_crop_size', type=int, nargs='*', default=[0, 0])
    parser.add_argument('--output_size', type=int, nargs='*', default=[224, 224])

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
    train_iter, valid_iter =\
        create_iterator(train, valid, mean, std,
                        args.pca_sigma, args.random_angle, args.x_random_flip,
                        args.y_random_flip, args.expand_ratio, args.random_crop_size,
                        args.random_erase, args.output_size, args.batchsize)

    n_class = 10
    npz_filename = None
    teacher = create_model(args.teacher_file,
                           args.teacher_name,
                           npz_filename,
                           n_class,
                           args.layer)
    serializers.load_npz(
        args.teacher_model,
        teacher, path='updater/model:main/predictor/')
    teacher.to_gpu()
    fun_generate_soft = partial(generate_softlabel, model=teacher, mean=mean, std=std)
    if not os.path.exists(args.softlabels_path):
        save_softlabels(train, fun_generate_soft, args.softlabels_path)
    if not os.path.exists('val' + args.softlabels_path):
        save_softlabels(valid, fun_generate_soft, 'val' + args.softlabels_path)

    img_t, lab_t = concat_examples(train)
    soft_labels_t = np.load(args.softlabels_path)
    train_soft = tuple_dataset.TupleDataset(img_t, soft_labels_t, lab_t)

    img_v, lab_v = concat_examples(valid)
    soft_labels_v = np.load('val' + args.softlabels_path)
    valid_soft = tuple_dataset.TupleDataset(img_v, soft_labels_v, lab_v)

    if args.caffe_model_path:
        npz_filename = caffe2npz(args.caffe_model_path)
    model = create_model(args.model_file,
                         args.model_name,
                         npz_filename,
                         n_class,
                         args.layers)
    net = L.Classifier(model)

    evaluator = extensions.Evaluator(valid_iter, net, device=args.gpu_id)

    trainer = create_trainer(train_iter, net, args.gpu_id, args.initial_lr,
                             args.weight_decay, args.freeze_layer, args.small_lr_layers,
                             args.small_initial_lr, args.num_epochs_or_iter,
                             args.epoch_or_iter, args.save_dir)
    if args.load_path:
        chainer.serializers.load_npz(args.load_path, trainer)

    trainer_extend(trainer, net, evaluator, args.small_lr_layers,
                   args.lr_decay_rate, args.lr_decay_epoch,
                   args.epoch_or_iter, args.save_trainer_interval)
    trainer.run()

