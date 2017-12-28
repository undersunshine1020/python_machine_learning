#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @TIME     : 2017/12/27 16:42
# @Author  : Gyt
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


if __name__ == '__main__':
    z = np.arange(-7, 7, 0.1)
    phi_z = sigmoid(z)
    plt.plot(z, phi_z)
    plt.axvline(0.0, color='k')
    # plt.axhspan(0.0, 1.0, facecolor='1.0', alpha=10.0, ls='dotted')
    plt.axhline(y=0.0, ls='dotted', color='k')
    plt.axhline(y=1.0, ls='dotted', color='k')
    plt.axhline(y=0.5, ls='dotted', color='k')
    plt.yticks([0.0, 0.5, 1.0])
    plt.ylim(-0.1, 1.1)
    plt.xlabel('z')
    plt.ylabel('$\phi (z)$')
    plt.show()
