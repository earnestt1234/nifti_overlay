#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 15:00:05 2024

@author: earnestt1234
"""

import matplotlib.pyplot as plt
import numpy as np

from .image import Anatomy, Mask

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
        self.color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.print = print if self.verbose else lambda *args, **kwargs: None
        self.prev_shape = None

    @property
    def nrows(self):
        return len(self.planes) if not self.transpose else self.nslices

    @property
    def ncols(self):
        return  self.nslices if not self.transpose else len(self.planes)

    @property
    def figx(self):
        return self.ncols * 2 if self.figsize == 'automatic' else self.figsize[0]

    @property
    def figy(self):
        return self.nrows * 2 if self.figsize == 'automatic' else self.figsize[1]

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


    def add_anat(self, path, colormap='gist_gray', alpha=1,
                 scale_panel=False, drop_zero=False, vmin=None,
                 vmax=None):
        img = Anatomy(path=path, colormap=colormap, alpha=alpha,
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

    def _init_figure(self):
        self.print()
        self.print("Initializing figure:")
        self.print(f"  Shape: {self.nrows}, {self.ncols}")
        self.print(f"  Size: {self.figx} in., {self.figy} in.")
        self.print(f"  DPI: {self.dpi}")

        self.fig, self.axes = None, None
        self.fig, self.axes = plt.subplots(self.nrows, self.ncols, figsize=(self.figx, self.figy), dpi=self.dpi)
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
        for n, image in enumerate(self.images):
            self._plot_image(n, image)

    def _plot_image(self, index, image):

        n = index

        self.print()
        self.print( "--------------------------------------------------")
        self.print(f"IMAGE {n}")
        self.print( "--------------------------------------------------")

        self.check_mismatched_dimensions(image)

        self.print()
        self.print(f"Image path: {image.path}")
        self.print(f"Shape: {image.shape}")
        self.print(f"Image type: {image.__class__.__name__}")

        data = image.data

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

                self.print()
                self.print(f'Plotting panel [{i}, {j}]')
                self.print("Call:")
                for k, v in panel_args.items():
                    self.print(f"  {k}: {v}")

                image.plot_slice(**panel_args)
                ax.set_aspect(1)
                ax.axis('off')
                ax.set_facecolor(self.background)

        self.print()
        self.print("--------------------------------------------------")
        self.print()
        self.print("--------------------------------------------------")

    def plot(self, ):
        self._check_images()
        self._init_figure()
        self._main_plot_loop()

    def save(output):
        pass

    def show():
        pass