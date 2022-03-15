import numpy as np
import random


def add_noise_depth(image, level=0.1):
    # from DeepIM-PyTorch
    if len(image.shape) == 3:
        row, col, ch = image.shape
        noise_level = random.uniform(0, level)
        gauss = noise_level * np.random.randn(row, col)
        gauss = np.repeat(gauss[:, :, np.newaxis], ch, axis=2)
    else:  # 2
        row, col = image.shape
        noise_level = random.uniform(0, level)
        gauss = noise_level * np.random.randn(row, col)
    noisy = image + gauss
    return noisy
