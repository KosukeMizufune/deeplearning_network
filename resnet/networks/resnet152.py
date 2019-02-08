import chainer
from chainer import links as L


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
