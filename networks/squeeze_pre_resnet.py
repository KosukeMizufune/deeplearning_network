import collections

import chainer
from chainer import functions as F
from chainer import links as L
from chainer.serializers import npz
from chainer import initializers


class Fire(chainer.Chain):
    """Fire module in SqueezeNet
    This is Fire module by chainer. If you find this paper, you can access https://arxiv.org/abs/1602.07360.

    For initialization, you need 4 inputs:
    :param in_size: int, input channel size
    :param s1: int, output channel size in squeeze layer
    :param e1: int, output channel size in 1\times1 expand layer
    :param e3: int, output channel size in 3\times3 expand layer
    """

    def __init__(self, in_size, s1, e1, e3, bypass=False, initialW=None, initial_bias=None):
        super(Fire, self).__init__()
        self.bypass = bypass
        with self.init_scope():
            self.bn1 = L.BatchNormalization(in_size)
            self.bn2 = L.BatchNormalization(s1)
            self.squeeze1x1 = L.Convolution2D(in_size, s1, 1, initialW=initialW, initial_bias=initial_bias)
            self.expand1x1 = L.Convolution2D(s1, e1, 1, initialW=initialW, initial_bias=initial_bias)
            self.expand3x3 = L.Convolution2D(s1, e3, 3, pad=1, initialW=initialW, initial_bias=initial_bias)

    def __call__(self, x):
        h = F.relu(self.bn1(x))
        h = F.relu(self.bn2(self.squeeze1x1(h)))
        h_1 = self.expand1x1(h)
        h_3 = self.expand3x3(h)
        h_out = F.concat([h_1, h_3], axis=1)
        if self.bypass:
            h_out += x

        return h_out


class SqueezePreResNetBase(chainer.Chain):
    """Network of Squeezenet v1.1
    Detail of this network is on https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1.
    """

    def __init__(self, kwargs):
        super(SqueezePreResNetBase, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 96, 7, stride=2, **kwargs)
            self.fire2 = Fire(96, 16, 64, 64, **kwargs)
            self.fire3 = Fire(128, 16, 64, 64, bypass=True, **kwargs)
            self.fire4 = Fire(128, 32, 128, 128, **kwargs)
            self.fire5 = Fire(256, 32, 128, 128, bypass=True, **kwargs)
            self.fire6 = Fire(256, 48, 192, 192, **kwargs)
            self.fire7 = Fire(384, 48, 192, 192, bypass=True, **kwargs)
            self.fire8 = Fire(384, 64, 256, 256, **kwargs)
            self.fire9 = Fire(512, 64, 256, 256, bypass=True, **kwargs)
            self.conv10 = L.Convolution2D(512, 1000, 1, pad=1, **kwargs)
            self.bn = L.BatchNormalization(512)

    @property
    def functions(self):
        return collections.OrderedDict([
            ('conv1', [self.conv1]),
            ('pool1', [self._max_pooling_2d]),
            ('fire2', [self.fire2]),
            ('fire3', [self.fire3]),
            ('fire4', [self.fire4]),
            ('pool2', [self._max_pooling_2d]),
            ('fire5', [self.fire5]),
            ('fire6', [self.fire6]),
            ('fire7', [self.fire7]),
            ('fire8', [self.fire8]),
            ('pool3', [self._max_pooling_2d]),
            ('fire9', [self.fire9]),
            ('bn', [F.dropout, self.bn, F.relu]),
            ('conv10', [self.conv10]),
            ('gap', [self._global_average_pooling_2d]),
            ('prob', [F.softmax]),
        ])

    def __call__(self, x, layers=None):
        if layers is None:
            layers = ['prob']

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

    def _max_pooling_2d(self, x):
        return F.max_pooling_2d(x, 3, stride=2)

    def _global_average_pooling_2d(self, x):
        n, channel, rows, cols = x.data.shape
        h = F.average_pooling_2d(x, (rows, cols), stride=1)
        h = F.reshape(h, (n, channel))
        return h


class SqueezePreResNet(chainer.Chain):
    """Example of Squeezenet v1.1
    This is just a example of SqueezeNet1.1.
    You may sometimes change some layers (For example, you may change fine-tuning layers).

    :param n_out: int, the number of class
    :param pretrained_mdoel: str, pretrained model path
    :return Variable, batchsize \times n_out matrix
    """

    def __init__(self, n_out, pretrained_model=None, init_param=None):
        # setup
        if pretrained_model:
            # As a sampling process is time-consuming,
            # we employ a zero initializer for faster computation.
            kwargs = {'initialW': initializers.constant.Zero()}
        else:
            # employ default initializers used in the original paper
            kwargs = {'initialW': init_param}

        super(SqueezePreResNet, self).__init__()
        self.n_out = n_out

        with self.init_scope():
            self.base = SqueezePreResNetBase(kwargs)
            self.conv10 = L.Convolution2D(512, n_out, 1, pad=1, initialW=init_param)
        if pretrained_model:
            npz.load_npz(pretrained_model, self.base, strict=False)

    def __call__(self, x):
        h = self.base(x, layers=['bn'])['bn']
        h = F.relu(self.conv10(h))
        h = F.average_pooling_2d(h, h.array.shape[2])
        y = F.reshape(h, (-1, self.n_out))
        return y
