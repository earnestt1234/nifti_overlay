#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 13:47:54 2024

@author: earnestt1234
"""

import numpy as np

from nifti_overlay.image import Edges

def test_defaults(nifti_path):
    img = Edges(nifti_path)
    assert img.color == 'yellow'
    assert img.alpha == 1.0
    assert img.sigma == 1.0
    assert img.interpolation == 'none'

def test_get_slice(nifti_path):
    img = Edges(nifti_path)

    # x
    data = img.get_slice(0, 3)
    answ = np.array([[False, False, False, False, False, False, False],
                     [False,  True,  True,  True,  True,  True, False],
                     [False,  True, False, False, False,  True, False],
                     [False,  True,  True,  True,  True,  True, False],
                     [False, False, False, False, False, False, False]])
    assert np.array_equal(data, answ)

    # y
    data = img.get_slice(1, 3)
    answ = np.array([[False, False, False, False, False, False, False],
                     [False,  True,  True,  True,  True,  True, False],
                     [False,  True, False, False, False,  True, False],
                     [False,  True,  True,  True,  True,  True, False],
                     [False, False, False, False, False, False, False]])
    assert np.array_equal(data, answ)

    # z
    data = img.get_slice(2, 3)
    answ = np.array([[False, False, False, False, False, False, False],
                     [False,  True,  True,  True,  True,  True, False],
                     [False,  True, False, False, False,  True, False],
                     [False,  True, False, False, False,  True, False],
                     [False,  True, False, False, False,  True, False],
                     [False,  True,  True,  True,  True,  True, False],
                     [False, False, False, False, False, False, False]])
    assert np.array_equal(data, answ)
