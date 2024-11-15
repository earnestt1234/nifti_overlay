#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 20:20:28 2024

@author: earnestt1234
"""

import matplotlib
import numpy as np
from numpy import nan
import pytest

from nifti_overlay.image import Anatomy

def test_defaults(nifti_path):
    img = Anatomy(nifti_path)
    assert img.color == 'gist_gray'
    assert img.alpha == 1
    assert img.scale_panel == False
    assert img.drop_zero == False
    assert img.vmin is None
    assert img.vmax is None

def test_drop_zero(nifti_path):
    img = Anatomy(nifti_path, drop_zero=True)
    datax = img.get_slice(0, 3)
    ansx = np.array([[nan, nan, nan, nan, nan, nan, nan],
                     [nan, 1., 1., 1., 1., 1., nan],
                     [nan, 1., 2., 2., 2., 1., nan],
                     [nan, 1., 1., 1., 1., 1., nan],
                     [nan, nan, nan, nan, nan, nan, nan]])
    assert np.array_equal(datax, ansx, equal_nan=True)

def test_get_slice(nifti_path):
    img = Anatomy(nifti_path)

    # x dimension
    datax = img.get_slice(0, 3)
    ansx = np.array([[0., 0., 0., 0., 0., 0., 0.],
                     [0., 1., 1., 1., 1., 1., 0.],
                     [0., 1., 2., 2., 2., 1., 0.],
                     [0., 1., 1., 1., 1., 1., 0.],
                     [0., 0., 0., 0., 0., 0., 0.]])
    assert np.all(datax == ansx)

    # y dimension
    datay = img.get_slice(1, 3)
    ansy = ansx
    assert np.all(datay == ansy)

    dataz = img.get_slice(2, 3)
    ansz =np.array([[0., 0., 0., 0., 0., 0., 0.],
                    [0., 1., 1., 1., 1., 1., 0.],
                    [0., 1., 1., 1., 1., 1., 0.],
                    [0., 1., 1., 1., 1., 1., 0.],
                    [0., 1., 1., 1., 1., 1., 0.],
                    [0., 1., 1., 1., 1., 1., 0.],
                    [0., 0., 0., 0., 0., 0., 0.]])
    assert np.all(dataz == ansz)

@pytest.mark.parametrize('dimension', [0, 1, 2])
@pytest.mark.parametrize('position', [0, 1, 2])
def test_plot_slice(nifti_path, dimension, position):
    img = Anatomy(nifti_path)
    ax = img.plot_slice(dimension, position)
    plot_arr = ax.get_array()
    data_arr = img.get_slice(dimension, position)
    assert np.array_equal(plot_arr, data_arr)

def test_plot_slice_returns_ax(nifti_path):
    img = Anatomy(nifti_path)
    ax = img.plot_slice(0, 0)
    assert isinstance(ax, matplotlib.image.AxesImage)

def test_vmax(nifti_path):
    vmax = 0.5
    img = Anatomy(nifti_path, vmax=vmax)
    ax = img.plot_slice(0, 0)
    assert ax.get_clim()[1] == vmax

def test_vmin(nifti_path):
    vmin = 0.5
    img = Anatomy(nifti_path, vmin=vmin)
    ax = img.plot_slice(0, 0)
    assert ax.get_clim()[0] == vmin

def test_scale_panel(nifti_path):
    dim, pos = 0, 0
    img = Anatomy(nifti_path, scale_panel=True)
    data = img.get_slice(dim, pos)
    mini = np.nanmin(data)
    maxi = np.nanmax(data)
    ax = img.plot_slice(dim, pos)
    assert ax.get_clim() == (mini, maxi)


