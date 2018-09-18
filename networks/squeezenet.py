import chainer
from chainer import functions as F
from chainer import links as L


class Fire(chainer.Chain):
    """Fire module in SqueezeNet
    This is Fire module by chainer. If you find this paper, you can access https://arxiv.org/abs/1602.07360.

    For initialization, you need 4 inputs:
    :param in_size: int, input channel size
    :param s1: int, output channel size in squeeze layer
    :param e1: int, output channel size in 1\times1 expand layer
    :param e3: int, output channel size in 3\times3 expand layer
    """

    def __init__(self, in_size, s1, e1, e3):
        super(Fire, self).__init__()
        with self.init_scope():
            self.squeeze1x1 = L.Convolution2D(in_size, s1, 1)
            self.expand1x1 = L.Convolution2D(s1, e1, 1)
            self.expand3x3 = L.Convolution2D(s1, e3, 3, pad=1)

    def __call__(self, x):
        h = F.relu(self.squeeze1x1(x))
        h_1 = self.expand1x1(h)
        h_3 = self.expand3x3(h)
        h_out = F.concat([h_1, h_3], axis=1)

        return F.relu(h_out)


class SqueezeNetBase(chainer.Chain):
    """Network of Squeezenet v1.1
    Detail of this network is on https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1.
    
    :param n_out: int, the number of class
    """
    def __init__(self, n_out):
        super(SqueezeNetBase, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 64, 3, stride=2)
            self.fire2 = Fire(64, 16, 64, 64)
            self.fire3 = Fire(128, 16, 64, 64)
            self.fire4 = Fire(128, 32, 128, 128)
            self.fire5 = Fire(256, 32, 128, 128)
            self.fire6 = Fire(256, 48, 192, 192)
            self.fire7 = Fire(384, 48, 192, 192)
            self.fire8 = Fire(384, 64, 256, 256)
            self.fire9 = Fire(512, 64, 256, 256)
            self.conv10 = L.Convolution2D(512, n_out, 1, pad=1)

        self.n_out = n_out

    def __call__(self, x, train=False):
        with chainer.using_config('train', train):
            h = self.conv1(x)
            h = F.relu(h)
            h = F.max_pooling_2d(h, 3, stride=2)

            h = self.fire2(h)
            h = self.fire3(h)
            h = F.max_pooling_2d(h, 3, stride=2)

            h = self.fire4(h)

            h = self.fire5(h)
            h = F.max_pooling_2d(h, 3, stride=2)

            h = self.fire6(h)
            h = self.fire7(h)
            h = self.fire8(h)

            h = self.fire9(h)
            h = F.dropout(h, ratio=0.5)

            h = F.relu(self.conv10(h))
            h = F.average_pooling_2d(h, 13)
            y = F.reshape(h, (-1, self.n_out))

        return y
