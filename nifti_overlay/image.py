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

class Image(ABC):

    def __init__(self, path):
        self.path = path
        self.nifti = nib.load(self.path)
        self.nifti = nib.as_closest_canonical(self.nifti)
        self.data = self.nifti.get_fdata()

    @property
    def shape(self):
        return self.nifti.header.get_data_shape()

    @abstractmethod
    def plot_slice(self, dimension, position, ax=None, **kwargs):
        ...

class Anatomy(Image):

    def __init__(self, path, colormap='gist_gray', alpha=1,
                 scale_panel=False, drop_zero=False, vmin=None,
                 vmax=None):
        super().__init__(path)
        self.colormap = colormap
        self.alpha = alpha
        self.scale_panel = scale_panel
        self.drop_zero = drop_zero
        self.vmin = vmin
        self.vmax = vmax

    def plot_slice(self, dimension, position, ax=None, **kwargs):
        data = self.data.copy()

        if ax is None:
            ax = plt.gca()

        if self.drop_zero:
            data = np.where(data == 0, np.nan, data)
            data = np.ma.array(data, mask=np.isnan(data))

        xsect = np.rot90(np.take(data, indices=position, axis=dimension))

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
        ax.imshow(xsect, cmap=self.colormap,
                  aspect='auto', vmin=vmin, vmax=vmax,
                  alpha=self.alpha, **kwargs)

class Mask(Image):

    def __init__(self, path, color=None, alpha=1, mask_value=1):
        super().__init__(path)
        self.color = color
        self.alpha = alpha
        self.mask_value = mask_value

    def plot_slice(self, dimension, position, ax=None, _override_color=None, **kwargs):

        if ax is None:
            ax = plt.gca()

        if self.color is None and _override_color is None:
            raise ValueError('Either color attribute must be set, '
                             'or `_override_color` must be provided.')

        color = _override_color if _override_color else self.color

        data = np.where(self.data == self.mask_value, 1, np.nan)
        data = np.ma.array(data, mask=np.isnan(data))

        cmap = matplotlib.colors.ListedColormap(['black', color])
        bounds=[0,.5,2]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        xsect = np.rot90(np.take(data, indices=position, axis=dimension))
        ax.imshow(xsect, cmap=cmap, norm=norm, aspect='auto', alpha=self.alpha, **kwargs)
