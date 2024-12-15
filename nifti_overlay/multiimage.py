#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes for overlays computed from multiple images.  Currently, 
this is only for Checkerboards, but could be extended for other purposes
(e.g., plotting mathematical results derived from multiple images.)
"""

from abc import ABC, abstractmethod
import pathlib
from typing import Sequence
import warnings

import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from skimage.exposure.histogram_matching import match_histograms

from nifti_overlay.image import Anatomy

class MultiImage(ABC):
    """Class for plotting layers which are made up of multple images.
    **This is an abstract class and should not be used directly**."""

    def __init__(self, src: Sequence[str | pathlib.Path | nib.Nifti1Image]):
        """Instantiate a MultiImage object.

        Parameters
        ----------
        src : collection
            List of paths to images (string or pathlib) or Nibabel images.
        """
        self.src = src
        self.images = [Anatomy(p) for p in self.src]
        self.shape = self._determine_shape()

    def _determine_shape(self):
        """Determine the shape of the images contained within.  Will throw an error
        (ValueError) if there are no images or if the images have different 
        dimensions."""
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

    def aspect(self, dimension: int):
        """For a given dimension (int[0-2]), get the aspect ratio as indicated by the voxel size."""
        aspects = [img.aspect(dimension) for img in self.images]
        aspects_set = set(aspects)
        if len(aspects_set) != 1:
            warnings.warn(RuntimeWarning('Multiple aspects detected for MultiImage; using the one computed '
                                         'from the first image.'))

        return aspects[0]

    def dimension_shape(self, dimension: int):
        """For a given dimension (int[0-2]), get the size of the remaining 2 dimensions.
        When plotting over a given dimension/axis, the image shape will be what is returned here."""
        tmp = tuple((s for i, s in enumerate(self.shape) if i != dimension))
        rot90 = tmp[1], tmp[0]
        return rot90

    @abstractmethod
    def get_slice(self, dimension: int, position: int):
        """Plot a 2D slice from one axis (`dimension`) at a given depth (`position`)."""
        ...

    @abstractmethod
    def plot_slice(
        self,
        dimension: int,
        position: int,
        ax: None | matplotlib.axes.Axes = None, 
        **kwargs
        ):
        """Return the data for a 2D slice from one axis (`dimension`) at a given depth (`position`)."""
        ...

class CheckerBoard(MultiImage):
    """Multiple images plotted as an interleaved checkerboard."""

    def __init__(self, src: Sequence[str | pathlib.Path | nib.Nifti1Image],
                 boxes: int=10,
                 normalize: bool=True,
                 histogram_matching: bool=True,
                 color: str='gist_gray',
                 alpha: float=1.0):
        """Initialize a CheckerBoard.

        Parameters
        ----------
        src : Sequence[str, pathlib.Path, nib.Nifti1Image]
            Collection of images.
        boxes : int, optional
            The number of boxes across *in the shortest dimension of the slice being plotted*, by default 10
        normalize : bool, optional
            Normalize slices being plotted to be between 0 and 1, by default True
        histogram_matching : bool, optional
            Use `skimage.exposure.histogram_matching` to normalize the intensities of images being
            plotted, by default True.  As one colormap is applied to all images, this can
            be good to make sure all images are visible.  May not be necessary if the images are
            all coming in with the same dynamic range.
        color : str, optional
            Matplotlib colormap to use, by default 'gist_gray'
        alpha : float, optional
            Color opacity, by default 1.0
        """

        super().__init__(src)
        self.boxes = boxes
        self.normalize = normalize
        self.histogram_matching = histogram_matching
        self.color = color
        self.alpha = alpha

    def _assemble_checkerboard_mask(self, dimension: int):
        """Return an array which matches the shape of the 
        dimensions being plotted and is filled with integers
        indicating which images (as indexed) should be plotted
        in which pixels.

        Parameters
        ----------
        dimension : int
            Plotting dimension of the image (0 for x, 1 for y, 2 for z).

        Returns
        -------
        numpy array
            Checkerboard array

        :meta public:
        """
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

    @classmethod
    def _make_checker_array(self, x: int, y: int, width: int, levels: int):
        """Create an array representing a checkerboard with a given number of checkers.

        Parameters
        ----------
        x : int
            Number of checkers in the x direction (note: not the array size, unless width=1).
        y : int
            Number of checkers in the y direction (note: not the array size, unless width=1).
        width : int
            Width/height of each checker - i.e., the length of a single checker in units of numpy array cells.
        levels : int
            How many values should be tiled.  E.g., if levels=2, only the integer values `0` and `1` will 
            be included and tiled alternatively.  If levels=5, then the values `0` through `4` will be tiled.

        Returns
        -------
        numpy array
            Integer array with a checkerboard patterning.
        """
        base_pattern = np.indices((x, y)).sum(axis=0) % levels
        checkers = np.kron(base_pattern, np.ones((width, width)))
        return checkers

    def get_slice(self, dimension: int, position: int):
        """Return the data to be plotted for a given slice.

        Parameters
        ----------
        dimension : int
            Image dimension to plot over (0 for x, 1 for y, 2 for z).
        position : int
            Index of the slice to plot.

        Returns
        -------
        numpy array
            Imaging slice data.
        """
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

    def plot_slice(
            self,
            dimension: int,
            position: int,
            ax: None | matplotlib.axes.Axes = None,
            **kwargs
            ):
        """Plot the data for a given slice.

        Parameters
        ----------
        dimension : int
            Image dimension to plot over (0 for x, 1 for y, 2 for z).
        position : int
            Index of the slice to plot.
        ax : None | matplotlib.axes.Axes, optional
            Matplotlib axes to plot on, by default None, in which case
            ``plt.gca()`` is used.

        Returns
        -------
        matplotlib axes
            Axes after plotting.
        """

        xsect = self.get_slice(dimension, position)

        if ax is None:
            ax = plt.gca()

        aspect = self.aspect(dimension)
        return ax.imshow(xsect, cmap=self.color,
                         aspect=aspect, alpha=self.alpha, **kwargs)
