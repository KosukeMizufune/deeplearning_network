import numpy
import six

import chainer
from chainer.backends import cuda
from chainer import function
import chainer.functions as F
from chainer.functions.activation import log_softmax
from chainer import variable
from chainer.functions.evaluation import accuracy
from chainer import link
from chainer import reporter


class DistillClassifier(link.Chain):
    compute_accuracy = True

    def __init__(self, predictor,
                 lossfun_soft=None,
                 lossfun_hard=F.softmax_cross_entropy,
                 alpha=0.8,
                 t=1.0,
                 accfun=accuracy.accuracy,
                 label_key=-1):
        if not (isinstance(label_key, (int, str))):
            raise TypeError('label_key must be int or str, but is %s' %
                            type(label_key))

        super(DistillClassifier, self).__init__()
        self.lossfun_soft = lossfun_soft
        self.lossfun_hard = lossfun_hard
        self.accfun = accfun
        self.y = None
        self.loss = None
        self.accuracy = None
        self.label_key = label_key
        self.loss_soft = 0
        self.loss_hard = 0
        self.alpha = alpha
        self.T = t

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

        self.y = None
        self.loss = None
        self.accuracy = None
        soft_label = args[1]
        soft_label = F.softmax(soft_label / self.T)

        self.y = self.predictor(args[0], **kwargs)
        if self.lossfun_soft:
            self.loss_soft = self.lossfun_soft(self.y / self.T, soft_label)
        self.loss_hard = self.lossfun_hard(self.y, t)
        self.loss = (1-self.alpha) * self.loss_soft * self.T * self.T + self.alpha * self.loss_hard
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.accuracy = self.accfun(self.y, t)
            reporter.report({'accuracy': self.accuracy}, self)
        return self.loss


def _broadcast_to(array, shape):
    if hasattr(numpy, 'broadcast_to'):
        return numpy.broadcast_to(array, shape)
    dummy = numpy.empty(shape, array.dtype)
    return numpy.broadcast_arrays(array, dummy)[0]


def _check_class_weight_option(class_weight):
    if class_weight is not None:
        if class_weight.ndim != 1:
            raise ValueError('class_weight.ndim should be 1')
        if class_weight.dtype.kind != 'f':
            raise ValueError('The dtype of class_weight should be \'f\'')
        if isinstance(class_weight, variable.Variable):
            raise ValueError('class_weight should be a numpy.ndarray or '
                             'cupy.ndarray, not a chainer.Variable')


def _check_reduce_option(reduce):
    if reduce not in ('mean', 'no'):
        raise ValueError(
            "only 'mean' and 'no' are valid for 'reduce', but '%s' is "
            'given' % reduce)


def _check_input_values(x, soft_lab):
    # Extract the raw ndarray as Variable.__ge__ is not implemented.
    # We assume that t is already an ndarray.
    if isinstance(x, variable.Variable):
        x = x.data

    if isinstance(soft_lab, variable.Variable):
        soft_lab = soft_lab.data


class SoftmaxCrossEntropySoftlabel(function.Function):
    normalize = True
    y = None

    def __init__(self, normalize=True, cache_score=True, class_weight=None,
                 ignore_label=-1, reduce='mean'):
        self.normalize = normalize
        self.cache_score = cache_score
        _check_class_weight_option(class_weight)
        self.class_weight = class_weight
        self.ignore_label = ignore_label
        _check_reduce_option(reduce)
        self.reduce = reduce
        self._coeff = 0.

    def forward_cpu(self, inputs):
        x, soft_label = inputs
        if chainer.is_debug():
            _check_input_values(x, soft_label)

        log_y = log_softmax._log_softmax(x)
        if self.cache_score:
            self.y = numpy.exp(log_y)
        if self.class_weight is not None:
            shape = [1 if d != 1 else -1 for d in six.moves.range(x.ndim)]
            log_y *= _broadcast_to(self.class_weight.reshape(shape), x.shape)

        log_p = numpy.array([numpy.sum(log_y * soft_label)])
        if self.normalize:
            count = x.shape[1]
        else:
            count = len(x)
        self._coeff = 1.0 / max(count, 1)

        y = log_p.sum(keepdims=True) * (-self._coeff)
        return y.reshape(()),

    def forward_gpu(self, inputs):
        cupy = cuda.cupy
        x, soft_label = inputs
        if chainer.is_debug():
            _check_input_values(x, soft_label)

        log_y = log_softmax._log_softmax(x)
        if self.cache_score:
            self.y = cupy.exp(log_y)
        if self.class_weight is not None:
            shape = [1 if d != 1 else -1 for d in six.moves.range(x.ndim)]
            log_y *= cupy.broadcast_to(
                self.class_weight.reshape(shape), x.shape)
        if self.normalize:
            coeff = cupy.maximum(1, soft_label.shape[1])
        else:
            coeff = max(1, len(soft_label))
        self._coeff = cupy.divide(1.0, coeff, dtype=x.dtype)

        ret = - cupy.sum(soft_label * log_y) * self._coeff
        return ret,

    def backward_cpu(self, inputs, grad_outputs):
        x, soft_label = inputs
        gloss = grad_outputs[0]
        if x.size == 0:
            return numpy.zeros(x.shape, dtype=x.dtype), None
        if self.y is not None:
            y = self.y.copy()
        else:
            y = log_softmax._log_softmax(x)
            numpy.exp(y, out=y)
        gx = y
        gx -= soft_label
        if self.class_weight is not None:
            shape = [1 if d != 1 else -1 for d in six.moves.range(x.ndim)]
            c = _broadcast_to(self.class_weight.reshape(shape), x.shape)
            gx *= _broadcast_to(numpy.expand_dims(c, 1), gx.shape)
        gx *= gloss * self._coeff

        return gx, None

    def backward_gpu(self, inputs, grad_outputs):
        cupy = cuda.cupy
        x, soft_label = inputs
        if x.size == 0:
            return cupy.zeros(x.shape, dtype=x.dtype), None
        if self.y is not None:
            y = self.y
        else:
            y = log_softmax._log_softmax(x)
            cupy.exp(y, out=y)
        gloss = grad_outputs[0]
        coeff = gloss * self._coeff

        if self.class_weight is None:
            gx = (y - soft_label) * coeff
        else:
            gx = (y - soft_label) * self.class_weight * coeff
        return gx, None


def softmax_cross_entropy_softlabel(x, t, normalize=True, cache_score=True,
                                    class_weight=None, ignore_label=-1,
                                    reduce='mean'):
    return SoftmaxCrossEntropySoftlabel(normalize,
                                        cache_score,
                                        class_weight,
                                        ignore_label,
                                        reduce)(x, t)
