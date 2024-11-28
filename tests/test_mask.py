#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 22:41:58 2024

@author: earnestt1234
"""

import matplotlib
import numpy as np
from numpy import nan
import pytest

from nifti_overlay.image import Mask

def test_defaults(nifti_path):
    img = Mask(nifti_path)
    assert img.color == 'red'
    assert img.alpha == 1
    assert img.mask_value == 1

def test_mask_value(nifti_path):
    img = Mask(nifti_path)

    # mask == 1
    img.mask_value = 1
    data = img.get_slice(0, 3)
    answ = np.array([[nan, nan, nan, nan, nan, nan, nan],
                     [nan, 1., 1., 1., 1., 1., nan],
                     [nan, 1., nan, nan, nan, 1., nan],
                     [nan, 1., 1., 1., 1., 1., nan],
                     [nan, nan, nan, nan, nan, nan, nan]])
    assert np.array_equal(data, answ, equal_nan=True)

    # mask == 2
    img.mask_value = 2
    data = img.get_slice(0, 3)
    answ = np.array([[nan, nan, nan, nan, nan, nan, nan],
                     [nan, nan, nan, nan, nan, nan, nan],
                     [nan, nan, 1., 1., 1., nan, nan],
                     [nan, nan, nan, nan, nan, nan, nan],
                     [nan, nan, nan, nan, nan, nan, nan]])
    assert np.array_equal(data, answ, equal_nan=True)

    # mask == 42
    img.mask_value = 42
    data = img.get_slice(0, 3)
    answ = np.array([[nan, nan, nan, nan, nan, nan, nan],
                     [nan, nan, nan, nan, nan, nan, nan],
                     [nan, nan, nan, nan, nan, nan, nan],
                     [nan, nan, nan, nan, nan, nan, nan],
                     [nan, nan, nan, nan, nan, nan, nan]])
    assert np.array_equal(data, answ, equal_nan=True)

def test_no_color_provided(nifti_path):
    img = Mask(nifti_path, color=None)
    with pytest.raises(ValueError):
        img.plot_slice(0, 1)

def test_override_color(nifti_path):
    img = Mask(nifti_path, color=None)
    img.plot_slice(0, 1, _override_color='red')
    assert True

@pytest.mark.parametrize('dimension', [0, 1, 2])
@pytest.mark.parametrize('position', [0, 1, 2])
def test_plot_slice(nifti_path, dimension, position):
    img = Mask(nifti_path)
    ax = img.plot_slice(dimension, position)
    plot_arr = ax.get_array()
    data_arr = img.get_slice(dimension, position)
    assert np.array_equal(plot_arr, data_arr, equal_nan=True)

def test_plot_slice_returns_ax(nifti_path):
    img = Mask(nifti_path)
    ax = img.plot_slice(0, 0)
    assert isinstance(ax, matplotlib.image.AxesImage)


