#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:42:00 2024

@author: earnestt1234
"""

import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure.histogram_matching import match_histograms

def _assemble_checkerboard_mask(images, dimension, position, boxes):
    target_shape = images[0].get_slice(dimension, position).shape
    target_x, target_y = target_shape
    shortest_side = min(target_shape)
    box_width = shortest_side // boxes
    boxes_x = target_x // box_width
    boxes_y = target_y // box_width
    cboard = _checkerboard_array(boxes_x, boxes_y, box_width, len(images))

    checker_x, checker_y = cboard.shape
    pad_x_before, adjust = divmod(target_x - checker_x, 2)
    pad_x_after = pad_x_before + adjust
    pad_y_before, adjust = divmod(target_y - checker_y, 2)
    pad_y_after = pad_y_before + adjust
    padding = [(pad_x_before, pad_x_after), (pad_y_before, pad_y_after)]
    cboard_padded = np.pad(cboard, pad_width=padding, mode='edge')
    return cboard_padded

# https://stackoverflow.com/q/2169478/13386979
def _checkerboard_array(x, y, width, levels):
    base_pattern = np.indices((x, y)).sum(axis=0) % levels
    checkers = np.kron(base_pattern, np.ones((width, width)))
    return checkers

def _plot_checkerboard_same_colormap(images, cboard, dimension, position,
                                     normalize=True, histogram_matching=True,
                                     cmap='gist_gray', ax=None, **kwargs):
    target_shape = images[0].get_slice(dimension, position).shape
    plot_data = np.zeros(target_shape, dtype=float)
    base = None
    for i, img in enumerate(images):
        xsect = img.get_slice(dimension, position)

        # normalization: min-max
        if normalize:
            xsect = (xsect - xsect.min()) / (xsect.max() - xsect.min())

        # apply histogram matching if desired
        # first image is used as the "base"
        if base is None:
            base = xsect
        elif histogram_matching:
            xsect = match_histograms(xsect, base)

        checkered = np.where(cboard == i, xsect, 0)
        plot_data += checkered

    if ax is None:
        ax = plt.gca()
    ax.imshow(plot_data, cmap=cmap,
              aspect='auto', **kwargs)

# This may be implemented at some point, but for now, all
# checkerboards are plotted with one colormap

# def _plot_checkerboard_mixed_colormap(images, cboard, dimension, position):

#     target_shape = images[0].get_slice(dimension, position).shape
#     plot_data = np.zeros(target_shape, dtype=float)


def plot_checkerboard(images, dimension, position, boxes=10,
                      normalize=True, histogram_matching=True,
                      cmap='gist_gray', ax=None, **kwargs):
    cboard = _assemble_checkerboard_mask(images=images,
                                         dimension=dimension,
                                         position=position,
                                         boxes=boxes)

    _plot_checkerboard_same_colormap(images=images,
                                     cboard=cboard,
                                     dimension=dimension,
                                     position=position,
                                     normalize=normalize,
                                     histogram_matching=histogram_matching,
                                     cmap=cmap,
                                     ax=ax,
                                     **kwargs)

