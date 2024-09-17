#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 15:00:05 2024

@author: earnestt1234
"""

from itertools import cycle
import os

import matplotlib.pyplot as plt
import numpy as np

from nifti_overlay.image import Anatomy, Mask

class NiftiOverlay:

    def __init__(self, planes='xyz', nslices=7, transpose=False, min_all=None,
                 max_all=None, minx=0.15, maxx=0.85, miny=0.15, maxy=0.85,
                 minz=0.15, maxz=0.85, background='black',
                 figsize='automatic', dpi=200, verbose=False):

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
        self.figure = None
        self.axes = None

        # holder for images to be plotted
        self.images = []

        # other variables
        self.planes_to_idx = {'x': 0, 'y': 1, 'z': 2}
        self.color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        self.print = print if self.verbose else lambda *args, **kwargs: None
        self.prev_shape = None
        self.automatic_figsize_scale = 1.0

    @property
    def nrows(self):
        return len(self.planes) if not self.transpose else self.nslices

    @property
    def ncols(self):
        return  self.nslices if not self.transpose else len(self.planes)

    @property
    def paddings(self):

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


    def add_anat(self, path, color='gist_gray', alpha=1,
                 scale_panel=False, drop_zero=False, vmin=None,
                 vmax=None):
        img = Anatomy(path=path, color=color, alpha=alpha,
                      scale_panel=scale_panel, drop_zero=drop_zero,
                      vmin=vmin, vmax=vmax)
        self.images.append(img)
        return img

    def add_mask(self, path, color=None, alpha=1, mask_value=1):
        img = Mask(path=path, color=color, alpha=alpha, mask_value=mask_value)
        self.images.append(img)
        return img

    def _check_images(self):
        if not self.images:
            raise RuntimeError('No images have been added for plotting. Use '
                               'NiftiOverlay().add_anat() or NiftiOverlay.add_mask() '
                               'to add images to be plotted')

    def _check_mismatched_dimensions(self, image):
        # check for mismatched dimensions
        shape = image.nifti.header.get_data_shape()
        shapex, shapey, shapez = shape
        if self.prev_shape and shape != self.prev_shape:
            raise ValueError(f"DIMENSION ERROR.  New image dimensions {shape} do not equal previously "
                             f"plotted dimensions {self.prev_shape}")
        self.prev_shape = shape

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
            x_ = len(self.axes) if not self.transpose else 1
            y_ = 1 if not self.transpose else len(self.axes)
            self.axes = self.axes.reshape([x_, y_])

        if self.transpose:
            self.axes = self.axes.T

    def _main_plot_loop(self):
        self.prev_shape = None
        total = len(self.images)
        for index, image in enumerate(self.images):
            self._plot_image(image, index, total)

    def _plot_image(self, image, index, total):

        n = index

        self.print()
        self.print( "--------------------------------------------------")
        self.print(f"IMAGE {n+1} / {total}")
        self.print( "--------------------------------------------------")

        self._check_mismatched_dimensions(image)

        self.print()
        self.print(f"Image path: {image.path}")
        self.print(f"Shape: {image.shape}")
        self.print(f"Image type: {image.__class__.__name__}")

        data = image.data
        total_panels = self.nrows * self.ncols
        mask_color = None
        if isinstance(image, Mask) and image.color is None:
            mask_color = next(self.color_cycle)

        for i, p in enumerate(self.planes):
            dimension = self.planes_to_idx[p]
            dimension_size = data.shape[dimension]
            min_window, max_window = self.paddings[p]
            min_slice = int(min_window*dimension_size)
            max_slice = int(max_window*dimension_size)
            num = self.nrows if self.transpose else self.ncols

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

                ax = self.axes[i, j]
                position = int(idx)

                panel_args = {'dimension': dimension, 'position': position, 'ax': ax}

                percentage = round(((i * len(indices) + j) / total_panels) * 100, 2)

                self.print()
                self.print(f'Plotting panel [{i}, {j}] ({percentage}%)')
                self.print("Call:")
                for k, v in panel_args.items():
                    self.print(f"  {k}: {v}")

                if mask_color is not None:
                    panel_args['_override_color'] = mask_color
                image.plot_slice(**panel_args)
                ax.set_aspect(1)
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
        self._init_figure()
        self._main_plot_loop()
        return self.fig

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
