import chainer
from chainer import link, reporter
import chainer.functions as F
from chainer.functions.evaluation import accuracy


class AttentionTransfer(link.Chain):
    compute_accuracy = True

    def __init__(self, predictor,
                 teacher,
                 lossfun_hard=F.softmax_cross_entropy,
                 beta=1e+3,
                 label_key=-1,
                 accfun=accuracy.accuracy):
        super(AttentionTransfer, self).__init__()
        self.label_key = label_key
        self.lossfun_hard = lossfun_hard
        self.loss = None
        self.beta = beta
        self.y = None
        self.teacher = teacher
        self.accfun = accfun
        self.accuracy = None
        with self.init_scope():
            self.predictor = predictor

    def forward(self, *args, **kwargs):
        if isinstance(self.label_key, int):
            if not (-len(args) <= self.label_key < len(args)):
                msg = 'Label key %d is out of bounds' % self.label_key
                raise ValueError(msg)
            t = args[self.label_key]
            if self.label_key == -1:
                args = args[:-1]
            else:
                args = args[:self.label_key] + args[self.label_key + 1:]
        elif isinstance(self.label_key, str):
            if self.label_key not in kwargs:
                msg = 'Label key "%s" is not found' % self.label_key
                raise ValueError(msg)
            t = kwargs[self.label_key]
            del kwargs[self.label_key]
        else:
            raise ValueError("Invalid type: label_key")

        # y_t = args[1]
        with chainer.using_config('train', False), \
                chainer.using_config('enable_backprop', False):
            _, g_t = self.teacher(args[0])
        y_s, g_s = self.predictor(args[0], **kwargs)
        attention_pair = [('res2', 'fire3'), ('res3', 'fire5'), ('res4', 'fire9')]
        loss_at = [at_loss(g_t[t_layer], g_s[s_layer]) for t_layer, s_layer in attention_pair]
        self.loss = self.lossfun_hard(y_s, t) + self.beta / 2 * sum(loss_at)
        self.y = y_s
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.accuracy = self.accfun(self.y, t)
            reporter.report({'accuracy': self.accuracy}, self)
        return self.loss


def at_loss(x, y):
    return F.mean((at(x) - at(y)) ** 2)


def at(x):
    return F.normalize(F.mean(x**2, axis=1))
