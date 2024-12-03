#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 20:30:51 2024

@author: earnestt1234
"""

import os

import matplotlib.pyplot as plt
import pytest

from nifti_overlay import NiftiOverlay

def test_add(nifti_path):
    overlay = NiftiOverlay()
    overlay.add_anat(nifti_path)
    overlay.add_mask(nifti_path)
    overlay.add_edges(nifti_path)
    overlay.add_checkerboard([nifti_path, nifti_path])
    assert len(overlay.images) == 4

def test_figure_dimensions_automatic():
    overlay = NiftiOverlay(planes='xy', nslices=3, figsize='automatic')
    _ = getattr(overlay,'automatic_figsize_scale')

    # default scale
    figx, figy = overlay._get_figure_dimensions()
    assert figx == 3
    assert figy == 2

    # alter scale
    overlay.automatic_figsize_scale = 3.5
    figx, figy = overlay._get_figure_dimensions()
    assert figx == 3 * 3.5
    assert figy == 2 * 3.5

def test_figure_dimensions_fixed():
    overlay = NiftiOverlay(figsize=(42, 100))
    assert overlay._get_figure_dimensions() == (42, 100)

def test_generate_png(nifti_path, png_path):
    overlay = NiftiOverlay()
    overlay.add_anat(nifti_path)
    overlay.generate(png_path)
    assert os.path.isfile(png_path)

def test_generate_separate(nifti_path, directory_path):
    overlay = NiftiOverlay(planes='xy', nslices=2)
    overlay.add_anat(nifti_path)
    overlay.generate(directory_path, separate=True)
    for r in range(overlay.nrows):
        for c in range(overlay.ncols):
            assert os.path.exists(os.path.join(directory_path, f'panel_{r}x{c}.png'))

def test_init_figure():
    a = NiftiOverlay(planes='y', nslices=5)
    b = NiftiOverlay(planes='y', nslices=5, transpose=True)

    a._init_figure()
    assert a.fig is not None
    assert a.axes.shape == (1, 5)

    b._init_figure()
    assert b.fig is not None
    assert b.axes.shape == (5, 1)

def test_mask_color_cycle(nifti_path):
    overlay = NiftiOverlay()
    overlay.add_mask(nifti_path)
    overlay.plot()
    nextcolor = next(overlay.color_cycle)
    second_plt_color = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])[1]
    assert nextcolor == second_plt_color

def test_ncols():
    a = NiftiOverlay(planes='xyz', nslices=7, transpose=False)
    b = NiftiOverlay(planes='xyz', nslices=7, transpose=True)
    assert a.ncols == 7
    assert b.ncols == 3

def test_nrows():
    a = NiftiOverlay(planes='xyz', nslices=7, transpose=False)
    b = NiftiOverlay(planes='xyz', nslices=7, transpose=True)
    assert a.nrows == 3
    assert b.nrows == 7

def test_paddings():
    a = NiftiOverlay(minx=0.1, maxx=0.9,
                     miny=0.2, maxy=0.8,
                     minz=0.3, maxz=0.7,
                     min_all=None, max_all=None)
    b = NiftiOverlay(minx=0.1, maxx=0.9,
                     miny=0.2, maxy=0.8,
                     minz=0.3, maxz=0.7,
                     min_all=0.4, max_all=0.6)

    assert a.paddings['x'] == (0.1, 0.9)
    assert a.paddings['y'] == (0.2, 0.8)
    assert a.paddings['z'] == (0.3, 0.7)

    assert b.paddings['x'] == (0.4, 0.6)
    assert b.paddings['y'] == (0.4, 0.6)
    assert b.paddings['z'] == (0.4, 0.6)

def test_planes_to_idx():
    overlay = NiftiOverlay()
    p2i = overlay.planes_to_idx
    assert p2i['x'] == 0
    assert p2i['y'] == 1
    assert p2i['z'] == 2
    with pytest.raises(KeyError):
        p2i['t']

def test_plot_mismatched_dimensions(nifti_path, nifti_path_alt_shape):
    overlay = NiftiOverlay()
    overlay.add_anat(nifti_path)
    overlay.add_anat(nifti_path_alt_shape)
    with pytest.raises(RuntimeError):
        overlay.plot()

def test_plot_no_images():
    overlay = NiftiOverlay()
    with pytest.raises(RuntimeError):
        overlay.plot()
