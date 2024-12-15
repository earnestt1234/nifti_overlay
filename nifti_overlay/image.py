#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes for overlays generating plot data from single image files.
"""

from abc import ABC, abstractmethod
import pathlib

import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage import feature

class Image(ABC):
    """Generic, abstract class for representing a single image to plot.
    **This is an abstract class, and should not be used directly.**"""

    def __init__(self, src: str | pathlib.Path | nib.Nifti1Image):
        """

        Parameters
        ----------
        src : str path, pathlib.Path, or nibabel.Nifti1Image
            Source image.

        Raises
        ------
        TypeError
            `src` is not an accepted type.
        ValueError
            Input image is not 3D.

        Returns
        -------
        None.

        """

        # read different formats
        if isinstance(src, str):
            self.nifti = nib.load(src)
            self.path = src
        elif isinstance(src, pathlib.Path):
            path = str(src)
            self.nifti = nib.load(path)
            self.path = path
        elif isinstance(src, nib.Nifti1Image):
            self.nifti = src
            self.path = None
        else:
            raise TypeError('Acceptable types for initialization are '
                            'str, pathlib.Path, or nibabel.Nifti1Image, '
                            f'not {type(src)}')

        # set other attributes
        self.nifti = nib.as_closest_canonical(self.nifti)
        self.data = self.nifti.get_fdata()

        # check shape
        if len(self.shape) != 3:
            raise ValueError(f'Image must be 3D; image dimensions are: {self.shape}')

    @property
    def shape(self):
        """Tuple specifying the length of each dimension."""
        return self.nifti.header.get_data_shape()

    def aspect(self, dimension: int):
        """For a given dimension (int[0-2]), get the aspect ratio as indicated by the voxel size."""
        voxel_dims = list(self.nifti.header["pixdim"][1:4])
        del voxel_dims[dimension]
        x, y = voxel_dims
        return y/x

    def dimension_shape(self, dimension: int):
        """For a given dimension (int[0-2]), get the size of the remaining 2 dimensions.
        When plotting over a given dimension/axis, the image shape will be what is returned here."""
        tmp = list(self.shape)
        del tmp[dimension]
        rot90 = tmp[1], tmp[0]
        return rot90

    @abstractmethod
    def plot_slice(
        self,
        dimension: int,
        position: int,
        ax: matplotlib.axes.Axes | None,
        **kwargs):
        """Plot a 2D slice from one axis (`dimension`) at a given depth (`position`)."""
        ...

    @abstractmethod
    def get_slice(self, dimension: int, position: int):
        """Return the data for a 2D slice from one axis (`dimension`) at a given depth (`position`)."""
        ...

class Anatomy(Image):
    """Class for plotting anatomical images, where the voxel values are typically
    continuous and should be plotted with a colormap."""

    def __init__(
            self,
            src: str | pathlib.Path | nib.Nifti1Image,
            color: str = 'gist_gray',
            alpha: float = 1.0,
            scale_panel: bool = False,
            drop_zero: bool = False,
            vmin: float | None = None,
            vmax: float | None = None
            ):
        """
        Instantiate an Anatomy image.

        Parameters
        ----------
        src : str | pathlib.Path | nib.Nifti1Image
            Source image.
        color : str, optional
            Colormap key from matplotlib. The default is 'gist_gray'.
        alpha : float, optional
            Color transparency. The default is 1.0.
        scale_panel : bool, optional
            When plotting, scale the colormap to be the extent of intensities observed
            in each individual panel, rather than across the whole 3D volume.
            Can be used to maximize the dynamic range within each panel, but
            makes comparisons across panels more difficult. The default is False.
        drop_zero : bool, optional
            Don't plot voxels which have a value of zero. The default is False.
        vmin : float | None, optional
            Bottom limit of color range. The default is None.
        vmax : float | None, optional
            Top limit of color range. The default is None.

        Returns
        -------
        None.

        """
        super().__init__(src)
        self.color = color
        self.alpha = alpha
        self.scale_panel = scale_panel
        self.drop_zero = drop_zero
        self.vmin = vmin
        self.vmax = vmax

    def get_slice(self, dimension: int, position: int):
        """Return the data to be plotted for a given slice. The data returned are the
        raw intensities of a slice through the 3D volume. 
        If ``drop_zero`` is specified, the zero values with be replaced with NaN.

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

        data = self.data

        if self.drop_zero:
            data = np.where(data == 0, np.nan, data)
            data = np.ma.array(data, mask=np.isnan(data)).data

        xsect = np.rot90(np.take(data, indices=position, axis=dimension))
        return xsect

    def plot_slice(self, dimension: int, position: int, ax: matplotlib.axes.Axes | None = None, **kwargs):
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
        aspect = self.aspect(dimension)
        return ax.imshow(xsect, cmap=self.color,
                         aspect=aspect, vmin=vmin, vmax=vmax,
                         alpha=self.alpha, **kwargs)

class Edges(Image):
    """Class for plotting the contours/edges (automatically detected)
    of a slice."""

    def __init__(
            self,
            src: str | pathlib.Path | nib.Nifti1Image,
            color: str='yellow',
            alpha: float=1.0,
            sigma: float=1.0,
            interpolation: str='none'
            ):
        """
        Instantiate an Edges image.

        Parameters
        ----------
        src : str path, pathlib.Path, or nibabel.Nifti1Image
            Source image.
        color : str, RGB, RGBA, optional
            Color understood by matplotlib. The default is 'yellow'.
        alpha : float, optional
            Color transparency. The default is 1.0.
        sigma : float, optional
            Standard deviation of the Gaussian filter of the Canny edge detector.
            The default is 1.0.  See `skimage.feature.canny`.
        interpolation : str, optional
            Type of interpolation for plotting images. The default is 'none'.
            See matplotlib for more details (`plt.imshow`).

        Returns
        -------
        None.

        """
        super().__init__(src)
        self.color = color
        self.alpha = alpha
        self.sigma = sigma
        self.interpolation = interpolation

    def get_slice(self, dimension: int, position: int):
        """Return the data to be plotted for a given slice.
        The data for a slice is first indexed, and then the Canny edge detector
        is run to generate the binary edge image for the slice.

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
        xsect = np.rot90(np.take(self.data, indices=position, axis=dimension))
        edges = feature.canny(xsect, sigma=self.sigma)
        return edges

    def plot_slice(self, dimension: int, position: int, ax: matplotlib.axes.Axes | None = None, **kwargs):
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

        if ax is None:
            ax = plt.gca()

        edges = self.get_slice(dimension, position)
        X = np.zeros(edges.shape + (4,))
        rgba = matplotlib.colors.to_rgba(self.color, alpha=self.alpha)
        X[edges] = rgba

        aspect = self.aspect(dimension)
        return ax.imshow(X, aspect=aspect, alpha=self.alpha, interpolation=self.interpolation, **kwargs)

class Mask(Image):
    """Class for plotting a binary mask, represented as a single color."""

    def __init__(
            self,
            src: str | pathlib.Path | nib.Nifti1Image,
            color: str = 'red',
            alpha: float = 1.0,
            mask_value: float = 1.0
            ):
        """
        Instantiate a Mask image.

        Parameters
        ----------
        src : str path, pathlib.Path, or nibabel.Nifti1Image
            Source image.
        color : str, RGB, RGBA, optional
            Color understood by matplotlib. The default is 'red'.
        alpha : float, optional
            Color transparency. The default is 1.0.
        mask_value : int or float, optional
            Value to be plotted. For segmentations with multiple labels,
            this can be used to specify the single label to be plotted.
            The default is 1.  (Note: to plot all values for a multi-label
            segmentation, you can plot it as an Anatomy instead of a Mask.)

        Returns
        -------
        None.

        """
        super().__init__(src)
        self.color = color
        self.alpha = alpha
        self.mask_value = mask_value

    def get_slice(self, dimension: int, position: int):
        """Return the data to be plotted for a given slice.
        The intesities for a slice are first extracted.  Then, all values
        matching the `mask_value` are replaced with `1`, and all other values
        are replaced with NaN.

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
        data = np.where(self.data == self.mask_value, 1, np.nan)
        data = np.ma.array(data, mask=np.isnan(data)).data
        xsect = np.rot90(np.take(data, indices=position, axis=dimension))
        return xsect

    def plot_slice(
            self,
            dimension: int,
            position: int,
            ax: matplotlib.axes.Axes | None = None,
            _override_color: str | None = None,
            **kwargs):
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
        _override_color : str | None
            Set a color which is only used at time of plotting.  Mostly
            for internal purposes.

        Returns
        -------
        matplotlib axes
            Axes after plotting.
        """

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
        aspect = self.aspect(dimension)
        return ax.imshow(xsect, cmap=cmap, norm=norm, aspect=aspect, alpha=self.alpha, **kwargs)
