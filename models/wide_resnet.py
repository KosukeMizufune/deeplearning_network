import collections

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.serializers import npz


# https://github.com/mitmul/chainer-cifar10/blob/master/models/wide_resnet.py
class WideBasic(chainer.Chain):
    def __init__(self, n_input, n_output, stride, dropout):
        w = chainer.initializers.HeNormal()
        super(WideBasic, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                n_input, n_output, 3, stride, 1, nobias=True, initialW=w)
            self.conv2 = L.Convolution2D(
                n_output, n_output, 3, 1, 1, nobias=True, initialW=w)
            self.bn1 = L.BatchNormalization(n_input)
            self.bn2 = L.BatchNormalization(n_output)
            if n_input != n_output:
                self.shortcut = L.Convolution2D(
                    n_input, n_output, 1, stride, nobias=True, initialW=w)
        self.dropout = dropout

    def __call__(self, x):
        x = F.relu(self.bn1(x))
        h = F.relu(self.bn2(self.conv1(x)))
        if self.dropout:
            h = F.dropout(h)
        h = self.conv2(h)
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        return h + shortcut


class WideBlock(chainer.ChainList):
    def __init__(self, n_input, n_output, count, stride, dropout):
        super(WideBlock, self).__init__()
        self.add_link(WideBasic(n_input, n_output, stride, dropout))
        for _ in range(count - 1):
            self.add_link(WideBasic(n_output, n_output, 1, dropout))

    def __call__(self, x):
        for link in self:
            x = link(x)
        return x


class WideResNet(chainer.Chain):
    def __init__(
            self, widen_factor, depth, n_class, dropout=True):
        k = widen_factor
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        n_stages = [16, 16 * k, 32 * k, 64 * k]
        w = chainer.initializers.HeNormal()
        super(WideResNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                None, n_stages[0], 3, 1, 1, nobias=True, initialW=w)
            self.wide2 = WideBlock(n_stages[0], n_stages[1], n, 1, dropout)
            self.wide3 = WideBlock(n_stages[1], n_stages[2], n, 2, dropout)
            self.wide4 = WideBlock(n_stages[2], n_stages[3], n, 2, dropout)
            self.bn5 = L.BatchNormalization(n_stages[3])
            self.fc6 = L.Linear(n_stages[3], n_class, initialW=w)

    @property
    def functions(self):
        return collections.OrderedDict([
            ('conv1', [self.conv1]),
            ('wide2', [self.wide2]),
            ('wide3', [self.wide3]),
            ('wide4', [self.wide4]),
            ('bn5', [self.bn5, F.relu, _global_average_pooling_2d]),
            ('fc6', [self.fc6]),
        ])

    def __call__(self, x, layers=None):
        if layers is None:
            layers = ['fc6']

        h = x
        activations = {}
        target_layers = set(layers)
        for key, funcs in self.functions.items():
            if len(target_layers) == 0:
                break
            for func in funcs:
                h = func(h)
            if key in target_layers:
                activations[key] = h
                target_layers.remove(key)
        return activations


class WideResNet402(chainer.Chain):
    def __init__(self, n_class, pretrained_model_path=None, layers=None):
        super(WideResNet402, self).__init__()
        if layers:
            self.layers = layers
        else:
            self.layers = ['fc6']
        with self.init_scope():
            self.model = WideResNet(2, 40, n_class)
        if pretrained_model_path:
            npz.load_npz(pretrained_model_path, self.model)

    def __call__(self, x):
        y = self.model(x, layers=['fc6'])['fc6']
        return y


class WideResNet401(chainer.Chain):
    def __init__(self, n_class, pretrained_model_path=None, layers=None):
        super(WideResNet401, self).__init__()
        if layers:
            self.layers = layers
        else:
            self.layers = ['fc6']
        with self.init_scope():
            self.model = WideResNet(1, 40, n_class)
        if pretrained_model_path:
            npz.load_npz(pretrained_model_path, self.model)

    def __call__(self, x):
        y = self.model(x, layers=['fc6'])['fc6']
        return y


def _global_average_pooling_2d(x):
    _, _, rows, cols = x.shape
    h = F.average_pooling_2d(x, (rows, cols), stride=1)
    return h
