#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing code for generating images of NIFTI images overlaid on each other.
"""

from itertools import cycle
import os
import pathlib
from typing import Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from nifti_overlay.image import Anatomy, Edges, Mask
from nifti_overlay.multiimage import CheckerBoard, MultiImage

class NiftiOverlay:
    """The main tool of the package, which creates tiled plots of one or more NIFTI images.
    """

    def __init__(
            self,
            planes: str = 'xyz',
            nslices: int = 7,
            transpose: bool = False,
            min_all: None | float = None,
            max_all: None | float = None,
            minx: float = 0.15,
            maxx: float = 0.85,
            miny: float = 0.15,
            maxy: float = 0.85,
            minz: float = 0.15,
            maxz: float = 0.85,
            background: str = 'black',
            figsize: Tuple | str = 'automatic',
            dpi: int = 200,
            verbose: bool = False
            ):
        """Initialize a NiftiOverlay object.

        Parameters
        ----------
        planes : str, optional
            String specifying the number of rows being plotted, and the dimension of the
            image being plotted on each row.  The length of the string indicates the number
            of rows being plotted, and the specific characters indicate the image dimensions
            being plotted ("x", "y", or "z") and their order.  Characters can be repeated
            or omitted.
        nslices : int, optional
            Integer specifying the number of slices to plot per plane, by default 7.
        transpose : bool, optional
            Make the overlay have shape [slices, planes] instead of [planes, slices], by default False.
        min_all : None | float, optional
            Proportion specifying the minimum extent of the image dimension over which to sample slices,
            by default None, in which case the minx, miny, and minz parameters are used.
        max_all : None | float, optional
            Proportion specifying the maximum extent of the image dimension over which to sample slices,
            by default None, in which case the minx, miny, and minz parameters are used.
        minx : float, optional
            Minimum slice sampling limit (proportion) in the x dimension, by default 0.15
        maxx : float, optional
            Maximum slice sampling limit (proportion) in the x dimension, by default 0.85
        miny : float, optional
            Minimum slice sampling limit (proportion) in the y dimension, by default 0.15
        maxy : float, optional
            Maximum slice sampling limit (proportion) in the y dimension, by default 0.85
        minz : float, optional
            Minimum slice sampling limit (proportion) in the z dimension, by default 0.15
        maxz : float, optional
            Maximum slice sampling limit (proportion) in the z dimension, by default 0.85
        background : str, optional
            Color passed to matplotlib which sets the background color of panels, by default 'black'
        figsize : Tuple | str, optional
            Figure size, by default 'automatic', which provides a standard setting.  Otherwise,
            a 2-tuple can be passed specifying the figure dimensions in inches.
        dpi : int, optional
            Figure DPI, by default 200
        verbose : bool, optional
            Report progress while generating overlays, by default False
        """

        # user-supplied attributes
        self.planes = planes
        self.nslices = nslices
        self.transpose = transpose
        self.minx = minx
        self.miny = miny
        self.minz = minz
        self.maxx = maxx
        self.maxy = maxy
        self.maxz = maxz
        self.min_all = min_all
        self.max_all = max_all
        self.background = background
        self.figsize = figsize
        self.dpi = dpi
        self.verbose = verbose

        # matplotlib stuff
        self.fig = None
        self.axes = None

        # holder for images to be plotted
        self.images = []

        # other variables
        self.planes_to_idx = {'x': 0, 'y': 1, 'z': 2}
        self.color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        self.print = print if self.verbose else lambda *args, **kwargs: None
        self.automatic_figsize_scale = 1.0

    @property
    def nrows(self):
        """Number of rows of panels that will be plotted."""
        return len(self.planes) if not self.transpose else self.nslices

    @property
    def ncols(self):
        """Number of columns of panels that will be plotted."""
        return  self.nslices if not self.transpose else len(self.planes)

    @property
    def paddings(self):
        """Get the slice sampling extent in each dimension.

        Returns
        -------
        dict
            Dictionary with keys "x", "y", and "z".  For each, the
            value are the slice sampling limits provided as a 2-tuple
            (minimum, maximum).
        """

        minx = self.minx
        miny = self.miny
        minz = self.minz
        if self.min_all is not None:
            minx = self.min_all
            miny = self.min_all
            minz = self.min_all

        maxx = self.maxx
        maxy = self.maxy
        maxz = self.maxz
        if self.max_all is not None:
            maxx = self.max_all
            maxy = self.max_all
            maxz = self.max_all

        paddings = {'x':(minx, maxx), 'y':(miny, maxy), 'z': (minz, maxz)}

        return paddings


    def add_anat(
            self,
            src: str | pathlib.Path | nib.Nifti1Image,
            color='gist_gray',
            alpha=1,
            scale_panel=False,
            drop_zero=False,
            vmin=None,
            vmax=None
            ) -> Anatomy:
        """Add an anatomical image to be plotted.  This is typically
        a continuously value image which will be represented with a
        continuous colormap.

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
        nifti_overlay.image.Anatomy
            Object for managing the image being plotted.
        """
        img = Anatomy(src=src, color=color, alpha=alpha,
                      scale_panel=scale_panel, drop_zero=drop_zero,
                      vmin=vmin, vmax=vmax)
        self.images.append(img)
        return img

    def add_checkerboard(
            self,
            src: str | pathlib.Path | nib.Nifti1Image,
            boxes: int = 10,
            color: str = 'gist_gray',
            alpha: float = 1.0,
            normalize: bool = True,
            histogram_matching: bool = True
            ) -> CheckerBoard:
        """Add a checkerboard to be plotted.

        Parameters
        ----------
        src : Sequence[str, pathlib.Path, nib.Nifti1Image]
            Collection of images.
        boxes : int, optional
            The number of boxes across *in the shortest dimension of the slice being plotted*, by default 10
        color : str, optional
            Matplotlib colormap to use, by default 'gist_gray'
        alpha : float, optional
            Color opacity, by default 1.0
        normalize : bool, optional
            Normalize slices being plotted to be between 0 and 1, by default True
        histogram_matching : bool, optional
            Use `skimage.exposure.histogram_matching` to normalize the intensities of images being
            plotted, by default True.  As one colormap is applied to all images, this can
            be good to make sure all images are visible.  May not be necessary if the images are
            all coming in with the same dynamic range.
        

        Returns
        -------
        CheckerBoard
            Object for managing the image being plotted.
        """
        img = CheckerBoard(src=src, boxes=boxes, color=color,
                           alpha=alpha, normalize=normalize,
                           histogram_matching=histogram_matching)
        self.images.append(img)
        return img 

    def add_edges(
            self,
            src: str | pathlib.Path | nib.Nifti1Image,
            color: str='yellow',
            alpha: float=1.0, 
            sigma: float=1.0,
            interpolation: str='none',
            ) -> Edges:
        """Add an edges image to be plotted.  The contours/edges
        of each slice are automatically detected.

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
        Edges
            Object for managing the image being plotted.
        """
        img = Edges(src=src, color=color, alpha=alpha, sigma=sigma, interpolation=interpolation)
        self.images.append(img)
        return img

    def add_mask(
            self,
            src: str | pathlib.Path | nib.Nifti1Image,
            color: str='red',
            alpha: float=1.0, 
            mask_value: float=1.0,
            ) -> Mask:
        """Add a mask to be plotted; a binary image plotted with a single
        color.

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
        Mask
            Object for managing the image being plotted.
        """
        img = Mask(src=src, color=color, alpha=alpha, mask_value=mask_value)
        self.images.append(img)
        return img

    def _check_images(self):
        if not self.images:
            raise RuntimeError('No images have been added for plotting. Use '
                               'NiftiOverlay().add_anat() or NiftiOverlay.add_mask() '
                               'to add images to be plotted')

    def _check_mismatched_dimensions(self):
        # no images added
        if not self.images:
            return None

        # all same shape
        shapes = [i.shape for i in self.images]
        shapeset = set(shapes)
        if len(shapeset) == 1:
            return shapes[0]

        # different shape
        else:
            raise RuntimeError(f"DIMENSION ERROR.  Found different image dimensions for different images: {shapeset}")

    def _get_figure_dimensions(self):
        if self.figsize == 'automatic':
            figx, figy = self.ncols, self.nrows
            figx *= self.automatic_figsize_scale
            figy *= self.automatic_figsize_scale
        else:
            figx, figy = self.figsize

        return figx, figy

    def _init_figure(self):

        figx, figy = self._get_figure_dimensions()

        self.print()
        self.print("Initializing figure:")
        self.print(f"  Shape: {self.nrows}, {self.ncols}")
        self.print(f"  Size: {figx} in., {figy} in.")
        self.print(f"  DPI: {self.dpi}")

        self.fig, self.axes = None, None
        self.fig, self.axes = plt.subplots(self.nrows, self.ncols, figsize=(figx, figy), dpi=self.dpi)
        self.fig.subplots_adjust(0,0,1,1,0,0)
        self.fig.patch.set_facecolor(self.background)

        # reshape axes
        if type(self.axes) != np.ndarray:
            self.axes = np.array(self.axes).reshape((1,1))
        elif self.axes.ndim == 2:
            pass
        elif self.axes.ndim == 1:
            self.axes = self.axes.reshape([self.nrows, self.ncols])

    def _main_plot_loop(self):
        total = len(self.images)
        for index, image in enumerate(self.images):
            self._plot_image(image, index, total)

    def _plot_image(self, image, index, total):

        n = index
        if isinstance(image, MultiImage):
            path = f'< {len(image.images)}  image(s) >'
        else:
            path = image.path

        self.print()
        self.print( "--------------------------------------------------")
        self.print(f"IMAGE {n+1} / {total}")
        self.print( "--------------------------------------------------")

        self.print()
        self.print(f"Image path: {path}")
        self.print(f"Shape: {image.shape}")
        self.print(f"Image type: {image.__class__.__name__}")

        total_panels = self.nrows * self.ncols
        mask_color = None
        if isinstance(image, Mask) and image.color is None:
            mask_color = next(self.color_cycle)

        for i, p in enumerate(self.planes):
            dimension = self.planes_to_idx[p]
            dimension_size = image.shape[dimension]
            min_window, max_window = self.paddings[p]
            min_slice = int(min_window*dimension_size)
            max_slice = int(max_window*dimension_size)
            num = self.nslices

            if num == 1:
                indices = [int((max_slice + min_slice) / 2)]
            else:
                indices = np.linspace(min_slice, max_slice, num)

            self.print()
            self.print(f"Plotting row [{i}]")
            self.print(f"Axis = '{p}'")
            self.print(f"Minimum & Maximum extent: {min_window}, {max_window}")
            self.print(f"Slices to plot along dimension (pre-rounding): {list(indices)}")

            for j, idx in enumerate(indices):

                percentage = round(((i * len(indices) + j) / total_panels) * 100, 2)

                self.print()
                self.print(f'Plotting panel [{i}, {j}] ({percentage}%)')

                ax_x = i if not self.transpose else j
                ax_y = j if not self.transpose else i
                ax = self.axes[ax_x, ax_y]
                position = int(idx)
                panel_args = {'dimension': dimension, 'position': position, 'ax': ax}

                self.print("Call:")
                for k, v in panel_args.items():
                    self.print(f"  {k}: {v}")

                if mask_color is not None:
                    panel_args['_override_color'] = mask_color
                image.plot_slice(**panel_args)
                ax.axis('off')
                ax.set_facecolor(self.background)

        self.print()
        self.print("Finished.")

        self.print()
        self.print("--------------------------------------------------")
        self.print()
        self.print("--------------------------------------------------")

    def plot(self):
        self._check_images()
        self._check_mismatched_dimensions()
        self._init_figure()
        self._main_plot_loop()

    def generate(self, savepath, separate=False, rerun=True):

        if self.fig is None or rerun:
            self.plot()

        if not separate:
            self.print(f"Saving output to {savepath}...")
            self.fig.savefig(savepath, facecolor=self.background)
            return

        if not os.path.isdir(savepath):
            os.mkdir(savepath)
        for r in range(self.nrows):
            for c in range(self.ncols):
                ax = self.axes[r, c]
                extent = ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
                impath = os.path.join(savepath, f"panel_{r}x{c}.png")
                self.print(f"Saving panel at {impath}...")
                self.fig.savefig(impath, bbox_inches=extent, facecolor=self.background)
