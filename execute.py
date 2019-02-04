import chainer
from chainer import cuda, iterators, optimizers, training
from chainer.training import extensions, triggers


def run_train(train, valid, net, **kwargs):
    # Iterator
    train_iter = iterators.SerialIterator(train, kwargs['batchsize'])
    valid_iter = iterators.SerialIterator(valid, kwargs['batchsize'], repeat=False, shuffle=False)

    # Optimizer
    if kwargs['gpu_id'] >= 0:
        net.to_gpu(kwargs['gpu_id'])
    optimizer = optimizers.MomentumSGD(lr=kwargs['lr'])
    optimizer.setup(net)

    if kwargs['l2_lambda'] > 0:
        optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(kwargs['l2_lambda']))
    freeze_setup(net, optimizer, kwargs['freeze_layer'])

    if kwargs['changed_lr_layer']:
        for layer in kwargs['changed_lr_layer']:
            layer.update_rule.hyperparam.lr = kwargs['changed_lr']

    # Trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=kwargs['gpu_id'])
    trainer = training.Trainer(
        updater, (kwargs['max_epoch'], 'epoch'), out=kwargs['result_dir'])

    if kwargs['load_dir']:
        chainer.serializers.load_npz(kwargs['load_dir'], trainer)

    trainer_extend(trainer,
                   valid_iter,
                   net,
                   **kwargs)
    trainer.run()


def trainer_extend(trainer, valid_iter, net, **kwargs):
    def slow_drop_lr(trainer):
        if kwargs['changed_lr_layer'] is None:
            pass
        else:
            for layer in kwargs['changed_lr_layer']:
                layer.update_rule.hyperparam.lr *= kwargs['lr_drop_rate']

    trainer.extend(
        slow_drop_lr,
        trigger=triggers.ManualScheduleTrigger(kwargs['lr_drop_epoch'], 'epoch')
    )
    trainer.extend(extensions.ExponentialShift('lr', kwargs['lr_drop_rate']),
                   trigger=triggers.ManualScheduleTrigger(kwargs['lr_drop_epoch'], 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.observe_lr(), trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'),
                   trigger=(kwargs['save_trainer_interval'], 'epoch'))
    trainer.extend(extensions.Evaluator(valid_iter, net, device=kwargs['gpu_id']), name='val')
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'lr', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(
        extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    trainer.extend(extensions.dump_graph('main/loss'))


class DelGradient(object):
    name = 'DelGradient'

    def __init__(self, deltgt):
        self.deltgt = deltgt

    def __call__(self, opt):
        for name, param in opt.target.namedparams():
            for d in self.deltgt:
                if d in name:
                    grad = param.grad
                    with cuda.get_device(grad):
                        grad = 0


def freeze_setup(net, optimizer, freeze_layer):
    if freeze_layer == 'all':
        net.predictor.base.disable_update()
    elif isinstance(freeze_layer, list):
        optimizer.add_hook(DelGradient(freeze_layer))
    else:
        pass
