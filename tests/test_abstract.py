#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 21:03:25 2024

@author: earnestt1234
"""

import os

import nibabel as nib
from nifti_overlay.image import Image
import pytest

import numpy as np

class MockImage(Image):

    def get_slice(self):
        pass

    def plot_slice(self):
        pass

def test_access_data(nifti_path):
    img = MockImage(nifti_path)
    data = img.data
    assert isinstance(data, np.ndarray)
    assert data.shape == (7, 7, 5)

def test_access_nifti(nifti_path):
    img = MockImage(nifti_path)
    nifti = img.nifti
    assert isinstance(nifti, nib.Nifti1Image)

def test_access_path(nifti_path):
    img = MockImage(nifti_path)
    path = img.path
    assert isinstance(path, str)
    assert path.endswith('.nii.gz')
    assert os.path.exists(path)

@pytest.mark.parametrize('dimension', [0, 1, 2])
def test_dimension_shape(nifti_path, dimension):
    img = MockImage(nifti_path)
    shape = img.shape
    result = []
    for i in [0, 1, 2]:
        if dimension == i:
            continue
        result.append(shape[i])
    result = tuple(reversed(result))
    assert img.dimension_shape(dimension) == result

def test_dimension_shape_float(nifti_path):
    img = MockImage(nifti_path)
    with pytest.raises(TypeError):
        img.dimension_shape(1.)

def test_dimension_shape_oor(nifti_path):
    img = MockImage(nifti_path)
    with pytest.raises(IndexError):
        img.dimension_shape(5)

def test_load_non3d(nifti4d_path):
    with pytest.raises(ValueError):
        _ = MockImage(nifti4d_path)

def test_shape(nifti_path):
    img = MockImage(nifti_path)
    assert img.shape == (7, 7, 5)
