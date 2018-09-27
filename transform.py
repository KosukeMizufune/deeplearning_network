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


def transform(inputs, mean, std, output_size=(224, 224), x_random_flip=False, y_random_flip=False,
              random_crop_size=(0, 0), random_erase=False, train=False):
    x, lab = inputs
    x = x.copy()
    x -= mean[:, None, None]
    x /= std[:, None, None]
    x = x[::-1]
    if train:
        if x_random_flip or y_random_flip:
            x = transforms.random_flip(x, x_random=x_random_flip, y_random=y_random_flip)
        if all(random_crop_size) > 0:
            x = transforms.random_crop(x, random_crop_size)
        if random_erase:
            x = random_erasing(x)
    x = transforms.resize(x, output_size)
    return x, lab
