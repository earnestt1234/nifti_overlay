#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 22:41:58 2024

@author: earnestt1234
"""

import numpy as np
from numpy import nan

from nifti_overlay.image import Mask

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


