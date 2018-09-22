from functools import partial

import chainer
from chainer import iterators, optimizers, training
from chainer.training import extensions, triggers
import chainer.links as L
import numpy as np


def run_train(train, valid, model, batchsize=32, start_lr=0.001, lr_drop_ratio=0.1, lr_drop_epoch=20,
              l2_param=0, max_epoch=40, gpu_id=0):
    train_iter = iterators.SerialIterator(train, batchsize)
    valid_iter = iterators.SerialIterator(valid, batchsize, repeat=False, shuffle=False)

    net = L.Classifier(model)
    if gpu_id >= 0:
        net.to_gpu(gpu_id)

    optimizer = optimizers.MomentumSGD(lr=start_lr)
    optimizer.setup(net)
    if l2_param > 0:
        optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(l2_param))
    updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)

    trainer = training.Trainer(
        updater, (max_epoch, 'epoch'), out='cifar10_result')
    trainer_extend(trainer, net, lr_drop_ratio, lr_drop_epoch, valid_iter, gpu_id)
    trainer.run()


def trainer_extend(trainer, net, lr_drop_ratio, lr_drop_epoch, valid_iter, gpu_id):
    def lr_drop(trainer):
        trainer.updater.get_optimizer('main').lr *= lr_drop_ratio

    trainer.extend(
        lr_drop,
        trigger=triggers.ManualScheduleTrigger(lr_drop_epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
    trainer.extend(extensions.Evaluator(valid_iter, net, device=gpu_id), name='val')
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'conv1/W/data/std',
         'elapsed_time']))
    trainer.extend(extensions.ParameterStatistics(net.predictor.base.conv1, {'std': np.std}))
    trainer.extend(extensions.PlotReport(['conv1/W/data/std'], x_key='epoch', file_name='std.png'))
    trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(
        extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    trainer.extend(extensions.dump_graph('main/loss'))



