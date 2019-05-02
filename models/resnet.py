import chainer
from chainer import links as L
from chainer import initializers


class ResNet101(chainer.Chain):
    def __init__(self, n_out, pretrained_model='auto', layers=None):
        super(ResNet101, self).__init__()
        init_param = initializers.HeNormal()
        self.n_out = n_out
        if layers:
            self.layers = layers
        else:
            self.layers = ['pool5']

        with self.init_scope():
            self.base = L.ResNet101Layers(pretrained_model=pretrained_model)
            self.fc = L.Linear(None, n_out, initialW=init_param)

    def __call__(self, x):
        h = self.base(x, layers=self.layers)
        y = self.fc(h.pop(self.layers[-1]))
        if h:
            return y, h
        return y


class ResNet152(chainer.Chain):
    def __init__(self, n_out, init_param=None, pretrained_model='auto'):
        super(ResNet152, self).__init__()
        self.n_out = n_out

        with self.init_scope():
            self.base = L.ResNet152Layers(pretrained_model=pretrained_model)
            self.fc = L.Linear(None, n_out, initialW=init_param)

    def __call__(self, x):
        h = self.base(x, layers=['pool5'])['pool5']
        y = self.fc(h)
        return y
