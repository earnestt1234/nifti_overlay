#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 20:04:05 2024

@author: earnestt1234
"""

import pytest

from nifti_overlay.__main__ import (
    main,
    parse,
    parse_image_dict,
    parse_ordered_image_args
    )
from nifti_overlay.image import Anatomy, Edges
from nifti_overlay import NiftiOverlay

def test_default_arguments():
    test_args = parse()

    # Check default values for various arguments
    assert test_args.axes == 'xyz'
    assert test_args.nslices == 7
    assert test_args.dpi == 300
    assert test_args.background == 'black'

    # Check default values for boundary arguments
    assert test_args.min is None
    assert test_args.max is None
    assert test_args.minx == 0.15
    assert test_args.maxx == 0.85
    assert test_args.miny == 0.15
    assert test_args.maxy == 0.85
    assert test_args.minz == 0.15
    assert test_args.maxz == 0.85

def test_flag_arguments():
    test_args = parse(['-v', '-P', '-T'])
    assert test_args.verbose is True
    assert test_args.plot is True
    assert test_args.transpose is True

def test_generated_NiftiOverlay(nifti_path):
    output = main(
        arguments=['-A', nifti_path, '-c', 'jet',
                   '-M', nifti_path,
                   '-C', nifti_path, nifti_path, '--no-normalize', '--boxes', '17'],
        debug=True)
    assert isinstance(output, NiftiOverlay)
    assert len(output.images) == 3
    anat, mask, checker = output.images
    assert anat.path == nifti_path
    assert anat.color == 'jet'
    assert mask.path == nifti_path
    assert hasattr(mask, 'color')
    assert len(checker.images) == 2
    assert checker.normalize == False

def test_ordered_args_mechanism():

    # Test StoreValueInOrder
    test_args = parse(['-A', 'anatomy.nii', '-E', 'edges.nii'])
    assert hasattr(test_args, 'ordered_args')
    assert test_args.ordered_args == [
        ('anat', 'anatomy.nii'),
        ('edges', 'edges.nii')
    ]

    # Test StoreTrueInOrder
    test_args = parse(['-s', '-z'])
    assert hasattr(test_args, 'ordered_args')
    assert test_args.ordered_args == [
        ('scale_panel', True),
        ('drop_zero', True)
    ]

    # Test StoreFalseInOrder
    test_args = parse(['--no-normalize', '--no-matching'])
    assert hasattr(test_args, 'ordered_args')
    assert test_args.ordered_args == [
        ('normalize', False),
        ('histogram_matching', False)
    ]

@pytest.mark.filterwarnings("ignore")
def test_output_not_specified(nifti_path):
    main(['-A', nifti_path])
    assert True

def test_parse_image_dict(nifti_path):

    # basic
    d = {'type': 'anat',
         'src': nifti_path}
    img = parse_image_dict(d)
    assert isinstance(img, Anatomy)

    # with some extra arguments
    d = {'type': 'edges',
         'src': nifti_path,
         'color': 'jet'}
    img = parse_image_dict(d)
    assert isinstance(img, Edges)

    # wrong name of argumant
    d = {'type': 'mask',
         'src': nifti_path,
         'not_a_keyword': 'jet'}
    with pytest.raises(Exception):
        parse_image_dict(d)

    # bad image type
    d = {'type': 'pet',
         'src': nifti_path}
    with pytest.raises(Exception):
        parse_image_dict(d)

def test_parse_ordered_image_args():
    ordered_args=[
        ('anat', 'mri.nii.gz'),
        ('color', 'red'),
        ('mask', 'segmentation.nii.gz'),
        ('edges', 'overlay.nii.gz'),
        ('alpha', '1'),
        ('some_unrecognized_arg', 'haha'),
        ('checker', ['img1.nii.gz', 'img2.nii'])
        ]
    parsed = parse_ordered_image_args(ordered_args)
    assert len(parsed) == 4
    assert all('type' in image for image in parsed)
    assert all('src' in image for image in parsed)

def test_type_conversions():

    # Test float conversions
    test_args = parse(['--min', '1.0', '--max', '2.5'])
    assert test_args.min == 1.0
    assert test_args.max == 2.5

    # Test integer conversions
    test_args = parse(['--figx', '10', '--figy', '5'])
    assert test_args.figx == 10
    assert test_args.figy == 5




