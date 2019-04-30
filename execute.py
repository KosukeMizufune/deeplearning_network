import argparse

import chainer
from chainer import cuda, optimizers, training
from chainer.training import extensions, triggers
from chainer.optimizer_hooks import WeightDecay
from chainer import links as L

from utils import create_iterator, create_model, caffe2npz


def trainer_extend(trainer, net, evaluator, args):
    def slow_drop_lr(trainer):
        if args.small_lr_layers:
            for layer_name in args.small_lr_layers:
                layer = getattr(net.predictor, layer_name)
                layer.W.update_rule.hyperparam.lr *= args.lr_decay_rate
                layer.b.update_rule.hyperparam.lr *= args.lr_decay_rate

    # Learning rate
    trainer.extend(
        slow_drop_lr,
        trigger=triggers.ManualScheduleTrigger(args.lr_decay_epoch,
                                               args.epoch_or_iter)
    )
    trainer.extend(extensions.ExponentialShift('lr', args.lr_decay_rate),
                   trigger=triggers.ManualScheduleTrigger(args.lr_decay_epoch,
                                                          args.epoch_or_iter))

    # Observe training
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.observe_lr(), trigger=(1, args.epoch_or_iter))
    trainer.extend(evaluator, name='val')

    print_report = ["epoch",
                    "main/loss",
                    "main/accuracy",
                    "val/main/loss",
                    "val/main/accuracy",
                    "lr",
                    "elapsed_time"]
    trainer.extend(extensions.PrintReport(print_report))

    # save results of training
    trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'],
                                         x_key=args.epoch_or_iter,
                                         file_name='loss.png'))
    trainer.extend(
        extensions.PlotReport(['main/accuracy', 'val/main/accuracy'],
                              x_key=args.epoch_or_iter,
                              file_name='accuracy.png'))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(
        filename="snapshot_epoch-" + '{.updater.epoch}'),
        trigger=(args.save_trainer_interval, args.epoch_or_iter))


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


def run_train(train_iter, net, evaluator, args):
    # Optimizer
    if args.gpu_id >= 0:
        net.to_gpu(args.gpu_id)
    optimizer = optimizers.MomentumSGD(lr=args.initial_lr)
    optimizer.setup(net)

    if args.weight_decay > 0:
        optimizer.add_hook(WeightDecay(args.weight_decay))
    if args.freeze_layer:
        freeze_setup(net, optimizer, args.freeze_layer)

    if args.small_lr_layers:
        for layer_name in args.small_lr_layers:
            layer = getattr(net.predictor, layer_name)
            layer.W.update_rule.hyperparam.lr = args.small_initial_lr
            layer.b.update_rule.hyperparam.lr = args.small_initial_lr

    # Trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu_id)
    trainer = training.Trainer(
        updater, (args.num_epochs_or_iter, args.epoch_or_iter), out=args.save_dir)

    if args.load_path:
        chainer.serializers.load_npz(args.load_path, trainer)

    trainer_extend(trainer, net, evaluator, args)
    trainer.run()


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

    args = parser.parse_args()
    # TODO: small_lr_layers

    train_iter, valid_iter = create_iterator(args)

    npz_filename = None
    if args.caffe_model_path:
        npz_filename = caffe2npz(args.caffe_model_path)
    n_class = 10
    model = create_model(args.model_file,
                         args.model_name,
                         npz_filename,
                         n_class,
                         args.layers)
    net = L.Classifier(model)

    evaluator = extensions.Evaluator(valid_iter, net, device=args.gpu_id)
    run_train(train_iter, net, evaluator, args)
