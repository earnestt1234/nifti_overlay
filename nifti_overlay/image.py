#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 15:16:34 2024

@author: earnestt1234
"""

from abc import ABC, abstractmethod

import nibabel as nib
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
        return self.nifti.get_data_shape()

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
        self.scale_panel = False
        self.drop_zero = False
        self.vmin = None
        self.vmax = None

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
            vmin = xsect.min() if self.scale_panel else self.vmin

        if self.vmax:
            vmax = self.vmax
        else:
            vmax = xsect.max() if self.scale_panel else self.vmax

        # plot
        ax.imshow(xsect, cmap=self.cmap, aspect='auto', vmin=vmin, vmax=vmax, **kwargs)

class Mask(Image):

    def __init__(self, path, color=None, alpha=1, mask_value=1):
        super().__init__(path)
        self.color = color
        self.alpha = alpha
        self.mask_value = mask_value
