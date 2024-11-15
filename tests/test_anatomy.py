#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 20:20:28 2024

@author: earnestt1234
"""

import matplotlib
import numpy as np
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
def test_get_slice_dropzero(nifti_path, dimension):
    img = Anatomy(nifti_path, drop_zero=True)
    data = img.get_slice(dimension, 3)
    assert isinstance(data, np.ma.MaskedArray)

@pytest.mark.parametrize('dimension', [0, 1, 2])
@pytest.mark.parametrize('position', [0, 1, 2])
def test_plot_slice(nifti_path, dimension, position):
    img = Anatomy(nifti_path)
    ax = img.plot_slice(dimension, position)
    plot_arr = ax.get_array()
    data_arr = img.get_slice(dimension, position)
    assert np.array_equal(plot_arr, data_arr)

@pytest.mark.parametrize('color', ['gist_grey', 'jet'])
@pytest.mark.parametrize('alpha', [1, .5])
@pytest.mark.parametrize('scale_panel', [True, False])
@pytest.mark.parametrize('drop_zero', [True, False])
@pytest.mark.parametrize('vmin', [None, 0.5])
@pytest.mark.parametrize('vmax', [None, 0.5])
def test_plot_slice_against_init_parameters(nifti_path,
                                            color,
                                            alpha,
                                            scale_panel,
                                            drop_zero,
                                            vmin,
                                            vmax):
    img = Anatomy(nifti_path,
                  color=color,
                  alpha=alpha,
                  scale_panel=scale_panel,
                  drop_zero=drop_zero,
                  vmin=vmin,
                  vmax=vmax)
    dimension = 0
    position = 0
    ax = img.plot_slice(dimension, position)
    plot_arr = ax.get_array()
    data_arr = img.get_slice(dimension, position)
    assert np.array_equal(plot_arr, data_arr, equal_nan=True)

def test_plot_slice_returns_ax(nifti_path):
    img = Anatomy(nifti_path)
    result = img.plot_slice(0, 0)
    assert isinstance(result, matplotlib.image.AxesImage)

