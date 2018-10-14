import chainer
from chainer import cuda, iterators, optimizers, training
from chainer.training import extensions, triggers


def run_train(train, valid, net, batchsize=32, start_lr=0.001, lr_drop_ratio=0.1, lr_drop_epoch=20,
              partially_start_lr=0.001, partially_lr_drop_ratio=0.1, finetune_layer=None, freeze_layer=None,
              l2_param=0, max_epoch=40, gpu_id=0, result_dir='result', save_trainer_epoch=1, trainer_dir=None):
    # Iterator
    train_iter = iterators.SerialIterator(train, batchsize)
    valid_iter = iterators.SerialIterator(valid, batchsize, repeat=False, shuffle=False)

    # Optimizer
    if gpu_id >= 0:
        net.to_gpu(gpu_id)
    optimizer = optimizers.MomentumSGD(lr=start_lr)
    optimizer.setup(net)
    if l2_param > 0:
        optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(l2_param))
    freeze_setup(net, optimizer, freeze_layer)
    if finetune_layer:
        for layer in finetune_layer:
            layer.update_rule.hyperparam.lr = partially_start_lr

    # Trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)
    trainer = training.Trainer(
        updater, (max_epoch, 'epoch'), out=result_dir)
    if trainer_dir:
        chainer.serializers.load_npz(trainer_dir, trainer)
    trainer_extend(trainer, valid_iter, net, lr_drop_ratio, lr_drop_epoch,
                   partially_lr_drop_ratio, finetune_layer, gpu_id, save_trainer_epoch)
    trainer.run()


def trainer_extend(trainer, valid_iter, net, lr_drop_ratio, lr_drop_epoch,
                   partially_lr_drop_ratio, finetune_layer, gpu_id, save_trainer_epoch):
    def partially_drop_lr(trainer):
        if finetune_layer is None:
            pass
        else:
            for layer in finetune_layer:
                layer.update_rule.hyperparam.lr *= partially_lr_drop_ratio

    trainer.extend(
        partially_drop_lr,
        trigger=triggers.ManualScheduleTrigger(lr_drop_epoch, 'epoch')
    )
    trainer.extend(extensions.ExponentialShift('lr', lr_drop_ratio),
                   trigger=triggers.ManualScheduleTrigger(lr_drop_epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.observe_lr(), trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'),
                   trigger=(save_trainer_epoch, 'epoch'))
    trainer.extend(extensions.Evaluator(valid_iter, net, device=gpu_id), name='val')
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
