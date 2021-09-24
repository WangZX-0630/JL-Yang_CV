#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Image denoising module."""

from random import random
import time
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from util_args import *


def E_generator(beta, eta, h):
    """Generate energy function E and localized version of E.

    Usage: E, localized_E = E_generator(beta, eta, h)
    Formula:
        E = h * \sum{x_i} - beta * \sum{x_i x_j} - eta * \sum{x_i y_i}
    """
    def E(x, y):
        """Calculate energy for matrices x, y.

        Note: the computation is not localized, so this is quite expensive.
        """
        # sum of products of neighboring paris {xi, yi}
        xxm = np.zeros_like(x)
        xxm[:-1, :] = x[1:, :]  # down
        xxm[1:, :] += x[:-1, :]  # up
        xxm[:, :-1] += x[:, 1:]  # right
        xxm[:, 1:] += x[:, :-1]  # left
        xx = np.sum(xxm * x)
        xy = np.sum(x * y)
        xsum = np.sum(x)
        return h * xsum - beta * xx - eta * xy

    def is_valid(i, j, shape):
        """Check if coordinate i, j is valid in shape."""
        return i >= 0 and j >= 0 and i < shape[0] and j < shape[1]

    def localized_E(E1, i, j, x, y):
        """Localized version of Energy function E.

        Usage: old_x_ij, new_x_ij, E1, E2 = localized_E(Ecur, i, j, x, y)
        """
        oldval = x[i, j]
        newval = oldval * -1  # flip
        # local computations
        E2 = E1 - (h * oldval) + (h * newval)
        E2 = E2 + (eta * y[i, j] * oldval) - (eta * y[i, j] * newval)
        adjacent = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        neighbors = [x[i + di, j + dj] for di, dj in adjacent
                     if is_valid(i + di, j + dj, x.shape)]
        E2 = E2 + beta * sum(a * oldval for a in neighbors)
        E2 = E2 - beta * sum(a * newval for a in neighbors)
        return oldval, newval, E1, E2

    return E, localized_E


def temperature(k, kmax):
    """Schedule the temperature for simulated annealing."""
    return 1.0 / 500 * (1.0 / k - 1.0 / kmax)


def prob(E1, E2, t):
    """Probability transition function for simulated annealing."""
    return 1 if E1 > E2 else np.exp((E1 - E2) / t)


def simulated_annealing(y, kmax, E, localized_E, temp_dir):
    """Simulated annealing process for image denoising.

    Parameters
    ----------
    y: array_like
        The noisy binary image matrix ranging in {-1, 1}.
    kmax: int
        The maximun number of iterations.
    E: function
        Energy function.
    localized_E: function
        Localized version of E.
    temp_dir: path
        Directory to save temporary results.

    Returns
    ----------
    x: array_like
        The denoised binary image matrix ranging in {-1, 1}.
    energy_record:
        [time, Ebest] records for plotting.
    """
    x = np.array(y)
    Ebest = Ecur = E(x, y)  # initial energy
    initial_time = time.time()


    for k in range(1, kmax + 1):  # iterate kmax times
        start_time = time.time()
        t = temperature(k, kmax + 1)
        print ("k = %d, Temperature = %.4e" % (k, t))
        accept, reject = 0, 0
        for idx in np.ndindex(y.shape):  # for each pixel in the matrix
            old, new, E1, E2 = localized_E(Ecur, idx[0], idx[1], x, y)
            p, q = prob(E1, E2, t), random()
            if p > q:
                accept += 1
                Ecur, x[idx] = E2, new
                if (E2 < Ebest):
                    Ebest = E2  # update Ebest
            else:
                reject += 1
                Ecur, x[idx] = E1, old

        # record time and Ebest of this iteration
        end_time = time.time()


        print ("--- k = %d, accept = %d, reject = %d ---" % (k, accept, reject))
        print ("--- k = %d, %.1f seconds ---" % (k, end_time - start_time))

    return x


def denoise_image(image,dest):
    """Denoise a binary image.

    Usage: denoised_image, energy_record = denoise_image(image, args, method)
    """
    beta = 1e-3
    eta = 2e-4
    argh = 0.0
    kmax = 10
    data = sign(image.getdata(), {0: -1, 255: 1})  # convert to {-1, 1}
    E, localized_E = E_generator(beta, eta, argh)
    temp_dir = os.path.dirname(os.path.realpath(os.path.join('submits', 'img', dest)))
    y = data.reshape(image.size[::-1])  # convert 1-d array to matrix
    result = simulated_annealing(
    y, kmax, E, localized_E, temp_dir)
    result = sign(result, {-1: 0, 1: 255})
    output_image = Image.fromarray(result).convert('1', dither=Image.NONE)
    return output_image


def denoise_function():
    image_path = 'submits/compare/'
    for image in filter(lambda x: x.find('png')!=-1, os.listdir(image_path)):
        #args = get_args(src=image, dest=image)
        src=image
        dest = image
        # denoise and save result
        image = Image.open(os.path.join('submits', 'compare', src))
        result = denoise_image(image,dest)
        result.save(os.path.join('submits', 'img', dest))
        print ("[Saved]", os.path.join('submits', 'img', dest))

