import chainer
from chainer import links as L


class ResNet101(chainer.Chain):
    def __init__(self, n_out, init_param=None):
        super(ResNet101, self).__init__()
        self.n_out = n_out

        with self.init_scope():
            self.base = L.ResNet101Layers()
            self.fc = L.Linear(None, n_out, initialW=init_param)

    def __call__(self, x):
        h = self.base(x, layers=['pool5'])['pool5']
        y = self.fc(h)
        return y
