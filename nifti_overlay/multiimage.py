#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 09:38:48 2024

@author: earnestt1234
"""

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure.histogram_matching import match_histograms

from nifti_overlay.image import Anatomy

class MultiImage(ABC):

    def __init__(self, src):
        self.src = src
        self.images = [Anatomy(p) for p in self.src]
        self.shape = self._determine_shape()

    def _determine_shape(self):
        # no images added
        if not self.images:
            raise ValueError('At must one image must be provided for instantiation.')

        # all same shape
        shapes = [i.shape for i in self.images]
        shapeset = set(shapes)
        if len(shapeset) == 1:
            return shapes[0]

        # different shape
        else:
            raise ValueError(f"DIMENSION ERROR.  Found different image dimensions for different images: {shapeset}")

    def dimension_shape(self, dimension):
        tmp = tuple((s for i, s in enumerate(self.shape) if i != dimension))
        rot90 = tmp[1], tmp[0]
        return rot90

    @abstractmethod
    def get_slice(self, dimension, position):
        ...

    @abstractmethod
    def plot_slice(self, dimension, position, ax=None, **kwargs):
        ...

class CheckerBoard(MultiImage):

    def __init__(self, paths, boxes=10, normalize=True,
                 histogram_matching=True, color='gist_gray', alpha=1):
        super().__init__(paths)
        self.boxes = boxes
        self.normalize = normalize
        self.histogram_matching = histogram_matching
        self.color = color
        self.alpha = alpha

    def _assemble_checkerboard_mask(self, dimension):
        target_shape = self.dimension_shape(dimension)
        target_x, target_y = target_shape
        shortest_side = min(target_shape)
        box_width = shortest_side // self.boxes
        boxes_x = target_x // box_width
        boxes_y = target_y // box_width
        cboard = self._make_checker_array(boxes_x, boxes_y, box_width, len(self.images))

        checker_x, checker_y = cboard.shape
        pad_x_before, adjust = divmod(target_x - checker_x, 2)
        pad_x_after = pad_x_before + adjust
        pad_y_before, adjust = divmod(target_y - checker_y, 2)
        pad_y_after = pad_y_before + adjust
        padding = [(pad_x_before, pad_x_after), (pad_y_before, pad_y_after)]
        cboard_padded = np.pad(cboard, pad_width=padding, mode='edge')

        return cboard_padded

    def _make_checker_array(self, x, y, width, levels):
        base_pattern = np.indices((x, y)).sum(axis=0) % levels
        checkers = np.kron(base_pattern, np.ones((width, width)))
        return checkers

    def get_slice(self, dimension, position):
        cboard = self._assemble_checkerboard_mask(dimension)
        target_shape = self.dimension_shape(dimension)
        plot_data = np.zeros(target_shape, dtype=float)
        base = None
        for i, img in enumerate(self.images):
            xsect = img.get_slice(dimension, position)

            # normalization: min-max
            if self.normalize:
                xsect = (xsect - xsect.min()) / (xsect.max() - xsect.min())

            # apply histogram matching if desired
            # first image is used as the "base"
            if base is None:
                base = xsect
            elif self.histogram_matching:
                xsect = match_histograms(xsect, base)

            checkered = np.where(cboard == i, xsect, 0)
            plot_data += checkered

        return plot_data

    def plot_slice(self, dimension, position, ax=None, **kwargs):

        xsect = self.get_slice(dimension, position)

        if ax is None:
            ax = plt.gca()

        return ax.imshow(xsect, cmap=self.color,
                         aspect='auto', alpha=self.alpha, **kwargs)
