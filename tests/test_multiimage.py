#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 21:03:25 2024

@author: earnestt1234
"""

import pathlib

import nibabel as nib
from nifti_overlay.multiimage import MultiImage
import pytest

class MockImage(MultiImage):

    def get_slice(self):
        pass

    def plot_slice(self):
        pass

def test_abstract(multinifti_paths):
    with pytest.raises(TypeError):
        _ = MultiImage(multinifti_paths)

def test_load_frompath(multinifti_paths):
    img = MockImage(multinifti_paths)
    assert isinstance(img, MultiImage)

def test_load_frompathlib(multinifti_paths):
    img = MockImage([pathlib.Path(path) for path in multinifti_paths])
    assert isinstance(img, MultiImage)

def test_load_fromnibabel(multinifti_paths):
    img = MockImage([nib.load(path) for path in multinifti_paths])
    assert isinstance(img, MultiImage)

def test_load_mixed(multinifti_paths):
    a = multinifti_paths[0]
    b = pathlib.Path(multinifti_paths[1])
    c = nib.load(multinifti_paths[2])
    img = MockImage([a, b, c])
    assert isinstance(img, MultiImage)

def test_load_nonarray(nifti_path):
    with pytest.raises(Exception):
        _ = MockImage(nifti_path)

def test_load_noimages():
    with pytest.raises(ValueError):
        _ = MockImage([])

def test_load_mixedshapes(nifti_path, nifti_path_alt_shape):
    with pytest.raises(ValueError):
        _ = MockImage([nifti_path, nifti_path_alt_shape])

def test_shape(multinifti_paths):
    img = MockImage(multinifti_paths)
    assert img.shape == (7, 7, 5)
