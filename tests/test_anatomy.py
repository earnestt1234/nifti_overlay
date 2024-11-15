#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 20:20:28 2024

@author: earnestt1234
"""

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
