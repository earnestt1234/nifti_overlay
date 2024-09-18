#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 15:16:34 2024

@author: earnestt1234
"""

from abc import ABC, abstractmethod

import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage import feature

class Image(ABC):

    def __init__(self, path):
        self.path = path
        self.nifti = nib.load(self.path)
        self.nifti = nib.as_closest_canonical(self.nifti)
        self.data = self.nifti.get_fdata()

    @property
    def shape(self):
        return self.nifti.header.get_data_shape()

    def dimension_shape(self, dimension):
        tmp = tuple((s for i, s in enumerate(self.shape) if i != dimension))
        rot90 = tmp[1], tmp[0]
        return rot90

    @abstractmethod
    def plot_slice(self, dimension, position, ax=None, **kwargs):
        ...

    @abstractmethod
    def get_slice(self, dimension, position):
        ...

class Anatomy(Image):

    def __init__(self, path, color='gist_gray', alpha=1,
                 scale_panel=False, drop_zero=False, vmin=None,
                 vmax=None):
        super().__init__(path)
        self.color = color
        self.alpha = alpha
        self.scale_panel = scale_panel
        self.drop_zero = drop_zero
        self.vmin = vmin
        self.vmax = vmax

    def get_slice(self, dimension, position):
        data = self.data

        if self.drop_zero:
            data = np.where(data == 0, np.nan, data)
            data = np.ma.array(data, mask=np.isnan(data))

        xsect = np.rot90(np.take(data, indices=position, axis=dimension))
        return xsect

    def plot_slice(self, dimension, position, ax=None, **kwargs):
        data = self.data.copy()

        if ax is None:
            ax = plt.gca()

        xsect = self.get_slice(dimension, position)

        # set vmax/vmin
        if self.vmin:
            vmin = self.vmin
        else:
            vmin = xsect.min() if self.scale_panel else data.min()

        if self.vmax:
            vmax = self.vmax
        else:
            vmax = xsect.max() if self.scale_panel else data.max()

        # plot
        ax.imshow(xsect, cmap=self.color,
                  aspect='auto', vmin=vmin, vmax=vmax,
                  alpha=self.alpha, **kwargs)

class Edges(Image):

    def __init__(self, path, color='yellow', alpha=1.0, sigma=1.0, interpolation='none'):
        super().__init__(path)
        self.color = color
        self.alpha = alpha
        self.sigma = sigma
        self.interpolation = interpolation

    def get_slice(self, dimension, position):
        xsect = np.rot90(np.take(self.data, indices=position, axis=dimension))
        edges = feature.canny(xsect, sigma=self.sigma)
        return edges

    def plot_slice(self, dimension, position, ax=None, **kwargs):

        if ax is None:
            ax = plt.gca()

        edges = self.get_slice(dimension, position)
        X = np.zeros(edges.shape + (4,))
        rgba = matplotlib.colors.to_rgba(self.color, alpha=self.alpha)
        X[edges] = rgba

        ax.imshow(X, aspect='auto', alpha=self.alpha, interpolation=self.interpolation, **kwargs)

class Mask(Image):

    def __init__(self, path, color=None, alpha=1, mask_value=1):
        super().__init__(path)
        self.color = color
        self.alpha = alpha
        self.mask_value = mask_value

    def get_slice(self, dimension, position):
        data = np.where(self.data == self.mask_value, 1, np.nan)
        data = np.ma.array(data, mask=np.isnan(data))
        xsect = np.rot90(np.take(data, indices=position, axis=dimension))
        return xsect

    def plot_slice(self, dimension, position, ax=None, _override_color=None, **kwargs):

        if ax is None:
            ax = plt.gca()

        if self.color is None and _override_color is None:
            raise ValueError('Either color attribute must be set, '
                             'or `_override_color` must be provided.')

        color = _override_color if _override_color else self.color

        cmap = matplotlib.colors.ListedColormap(['black', color])
        bounds=[0,.5,2]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        xsect = self.get_slice(dimension, position)
        ax.imshow(xsect, cmap=cmap, norm=norm, aspect='auto', alpha=self.alpha, **kwargs)
