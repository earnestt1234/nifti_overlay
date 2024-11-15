#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 20:20:28 2024

@author: earnestt1234
"""

import nibabel as nib
import numpy as np

import pytest

AFFINE = np.array([[-1., 0., 0., 0],
                   [0., 1., 0., -126.,],
                   [0., 0., 1., -72.],
                   [0., 0., 0., 1.]])

def _make_nifti_4D():
    shape = (7, 7, 5, 3)
    data = np.random.rand(*shape)
    nii = nib.Nifti1Image(dataobj=data, affine=AFFINE)
    return nii

def _make_nifti_nested():
    shape = (7, 7, 5)
    data = np.zeros(shape)
    x, y, z = shape
    data[1:x-1, 1:x-1, 1:z-1] = 1
    data[2:x-2, 2:x-2, 2:z-2] = 2
    nii = nib.Nifti1Image(dataobj=data, affine=AFFINE)
    return nii

@pytest.fixture
def nifti4d_path(tmp_path_factory):
    nii = _make_nifti_4D()
    path = tmp_path_factory.mktemp("data") / "4d.nii.gz"
    nib.save(nii, path)
    return str(path)

@pytest.fixture
def nifti_path(tmp_path_factory):
    nii = _make_nifti_nested()
    path = tmp_path_factory.mktemp("data") / "nested.nii.gz"
    nib.save(nii, path)
    return str(path)
