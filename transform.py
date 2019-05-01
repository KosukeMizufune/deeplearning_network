from chainercv import transforms

import numpy as np
from skimage import transform as skimage_transform
import cv2 as cv

USE_OPENCV = False


def cv_rotate(img, angle):
    if USE_OPENCV:
        img = img.transpose(1, 2, 0) / 255.
        center = (img.shape[0] // 2, img.shape[1] // 2)
        r = cv.getRotationMatrix2D(center, angle, 1.0)
        img = cv.warpAffine(img, r, img.shape[:2])
        img = img.transpose(2, 0, 1) * 255.
        img = img.astype(np.float32)
    else:
        # scikit-image's rotate function is almost 7x slower than OpenCV
        img = img.transpose(1, 2, 0) / 255.
        img = skimage_transform.rotate(img, angle, mode='edge')
        img = img.transpose(2, 0, 1) * 255.
        img = img.astype(np.float32)
    return img


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


def transform_img(inputs, args, mean, std, train=False):
    x, lab = inputs
    x = x.copy()
    # Color augmentation
    if train and args.pca_sigma != 0:
        x = transforms.pca_lighting(x, args.pca_sigma)
    x -= mean[:, None, None]
    x /= std[:, None, None]
    x = x[::-1]
    if train:
        # Random rotate
        if args.random_angle != 0:
            angle = np.random.uniform(-args.random_angle, args.random_angle)
            x = cv_rotate(x, angle)

        # Random flip
        if args.x_random_flip or args.y_random_flip:
            x = transforms.random_flip(x, x_random=args.x_random_flip, y_random=args.y_random_flip)

        # Random expand
        if args.expand_ratio > 1:
            x = transforms.random_expand(x, max_ratio=args.expand_ratio)

        if all(args.random_crop_size) > 0:
            x = transforms.random_crop(x, args.random_crop_size)
        else:
            if args.random_erase:
                x = random_erasing(x)

    if all(args.random_crop_size) > 0:
        x = transforms.resize(x, args.random_crop_size)
    else:
        x = transforms.resize(x, args.output_size)

    return x, lab


def transform_with_softlabel(inputs, mean, std, train=False, **kwargs):
    x, soft_lab, lab = inputs
    x, lab = transform_img((x, lab), mean, std, train, **kwargs)
    return x, soft_lab, lab
