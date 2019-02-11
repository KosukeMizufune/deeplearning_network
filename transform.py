from chainercv import transforms

import numpy as np


def random_erasing(x, p=0.5, s_base=(0.02, 0.4), r_base=(0.3, 3)):
    x = x.copy()
    size = x.shape[2]
    if np.random.uniform(0, 1) > p:
        while True:
            s = np.random.uniform(s_base[0], s_base[1]) * size * size
            r = np.random.uniform(r_base[0], r_base[1])
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, size)
            top = np.random.randint(0, size)
            if left + w < size and top + h < size:
                break
        c = np.random.randint(0, 256)
        x[:, top:top + h, left:left + w] = c
    return x


def transform(inputs, mean, std, train=False, **kwargs):
    x, lab = inputs
    x = x.copy()
    x -= mean[:, None, None]
    x /= std[:, None, None]
    x = x[::-1]
    if train:
        if kwargs['x_random_flip'] or kwargs['y_random_flip']:
            x = transforms.random_flip(x, x_random=kwargs['x_random_flip'], y_random=kwargs['y_random_flip'])
        if all(kwargs['random_crop_size']) > 0:
            x = transforms.random_crop(x, kwargs['random_crop_size'])
        if kwargs['random_erase']:
            x = random_erasing(x)
    x = transforms.resize(x, kwargs['output_size'])
    return x, lab


def transform_with_softlabel(inputs, mean, std, train=False, **kwargs):
    x, soft_lab, lab = inputs
    x, lab = transform((x, lab), mean, std, train, **kwargs)
    return x, soft_lab, lab
