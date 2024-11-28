#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:18:48 2024

@author: earnestt1234
"""

import numpy as np

from nifti_overlay.multiimage import CheckerBoard

def test_assemble_checkerboard_mask(multinifti_paths):
    cb = CheckerBoard(multinifti_paths, boxes=2)
    arr = cb._assemble_checkerboard_mask(dimension=0)
    ans = np.array([[0., 0., 1., 1., 2., 2., 2.],
                    [0., 0., 1., 1., 2., 2., 2.],
                    [1., 1., 2., 2., 0., 0., 0.],
                    [1., 1., 2., 2., 0., 0., 0.],
                    [1., 1., 2., 2., 0., 0., 0.]])
    assert np.array_equal(arr, ans)
    assert ans.shape == cb.dimension_shape(0)

    cb = CheckerBoard(multinifti_paths[:2], boxes=2)
    arr = cb._assemble_checkerboard_mask(dimension=0)
    ans = np.array([[0., 0., 1., 1., 0., 0., 0.],
                    [0., 0., 1., 1., 0., 0., 0.],
                    [1., 1., 0., 0., 1., 1., 1.],
                    [1., 1., 0., 0., 1., 1., 1.],
                    [1., 1., 0., 0., 1., 1., 1.]])
    assert np.array_equal(arr, ans)
    assert ans.shape == cb.dimension_shape(0)


def test_defaults(multinifti_paths):
    img = CheckerBoard(multinifti_paths)
    assert img.boxes == 10
    assert img.normalize == True
    assert img.histogram_matching == True
    assert img.color == 'gist_gray'
    assert img.alpha == 1

def test_make_checker_array():
    # basic
    arr = CheckerBoard._make_checker_array(3, 3, 1, 2)
    ans = np.array([[0., 1., 0.],
                    [1., 0., 1.],
                    [0., 1., 0.]])
    assert np.array_equal(arr, ans)

    # alter number of checkers (x, y)
    arr = CheckerBoard._make_checker_array(2, 5, 1, 2)
    ans = np.array([[0., 1., 0., 1., 0.],
                    [1., 0., 1., 0., 1.]])
    assert np.array_equal(arr, ans)

    # alter width of checkers (width)
    arr = CheckerBoard._make_checker_array(3, 3, 2, 2)
    ans = np.array([[0., 0., 1., 1., 0., 0.],
                    [0., 0., 1., 1., 0., 0.],
                    [1., 1., 0., 0., 1., 1.],
                    [1., 1., 0., 0., 1., 1.],
                    [0., 0., 1., 1., 0., 0.],
                    [0., 0., 1., 1., 0., 0.]])
    assert np.array_equal(arr, ans)

    # alter number of different values (levels)
    arr = CheckerBoard._make_checker_array(3, 3, 1, 3)
    ans = np.array([[0., 1., 2.],
                    [1., 2., 0.],
                    [2., 0., 1.]])
    assert np.array_equal(arr, ans)

def test_normalize_histogram_matching(multinifti_paths):
    cb = CheckerBoard(multinifti_paths, boxes=2, normalize=True, histogram_matching=True)
    arr = cb.get_slice(0, 3)
    assert np.all(arr >= 0) and np.all(arr <= 1)



