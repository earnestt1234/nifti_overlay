#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:59:17 2024

@author: earnestt1234
"""

import argparse
import sys

import matplotlib
import matplotlib.pyplot as plt

from nifti_overlay.core import NiftiOverlay
from nifti_overlay.image import Anatomy, Mask

helptxt = """
------------------------
NIFTI Image Overlay Tool
------------------------

Author: Tom Earnest
Creation Date: 03-27-2022
GitHub: https://github.com/earnestt1234/nifti_overlay

Description
~~~~~~~~~~~

This script is a general tool for creating plots of NIFTI images.  It can be used
for plotting one or more anatomical / mask images in an array of automatically
determined slices.

Example uses:
    - Plotting an anatomical image.
    - Plotting a mask over an anatomical image to verify overlap.
    - Plotting the overlap of multiple masks.

Requirments
~~~~~~~~~~~
    - Python 3
        - matplotlib >= 3.2
        - nibabel >= 3.2
        - numpy >= 1.2
    - Inputs must be 3D NIFTI
    - Inputs have same dimension
    - Inputs do not have to have same orientation, but unexpected results may occur.

Usage
~~~~~

General:
    nifti_overlay ([-A anatomy] [anatomy options]) ([-M] [mask options]) [global options]

Plot an anatomical image:
    nifti_overlay -A t1.nii.gz -o my_img.png

Set the dimensions plotted and number of slices (only X & Y axes, 10 slices each):
    nifti_overlay -A t1.nii.gz -x 'xy' -n 10 -o my_img.png

Set the proportions of the image to plot:
    nifti_overlay -A t1.nii.gz -min .3 -max 1.0 -o my_img.png

Plot a mask over an anatomical image:
    nifti_overlay -A t1.nii.gz -M mask.nii.gz -c 'red' -a 0.3 -o my_img.png

Plot a PET image over a T1:
    nifti_overlay -A t1.nii.gz -A pet_registered.nii.gz -c jet -a 0.5 -o my_img.png

Plot an image in the default layout of FSLEYES:
    nifti_overlay -A img.nii.gz -T -n 1 -o my_img.png

Arguments: Image Specific
~~~~~~~~~~~~~~~~~~~~~~~~~

These arguments select images to plot and how.  As many of these can be provided
as images are to be plotted.  These are grouped according to their order,
such that IMAGE OPTIONS following an IMAGE will be assigned to that image.
The images are plotted in order, so later images will be "on top".

IMAGES

    -A / --anat ANATOMY
        Path to an anatomical image to plot

    -M / --mask MASK
        Path to a mask image to plot

IMAGE OPTIONS (ALL)

    -c / --color COLOR
        Color for the image.
            For an ANATOMY, this should be a matplotlib colormap (see https://matplotlib.org/3.5.1/tutorials/colors/colormaps.html)
            For a MASK, this should be a color name (see https://matplotlib.org/stable/gallery/color/named_colors.html)

    -a / --alpha ALPHA
        Color alpha.  Default: 1

IMAGE OPTIONS (ANATOMY)

    -s / --scalepanel
        Scale the colormap relative to the panel, rather than the entire image.

    -z / --dropzero
        Omit plotting voxels with 0 intensity.

    --vmin / --vmax VALUE
        Provide minimum or maximum values for the color map.

IMAGE OPTIONS (MASK)

    -m / --maskvalue MASKVALUE
        The value corresponding to the binary mask value. Default: 1.

Arguments: Global
~~~~~~~~~~~~~~~~~

These arguments affect all images.

    -h / --help
        Show this message and exit.

    -v / --verbose
        Add status print statements.

    -o / --output OUTPUT
        Path to save output image, with extension.  See matplotlib documentation
        for usuable extension.  If -S / --separate is passed, the output is instead
        a path to folder to save images within.

    -P / --plot OUTPUT
        Show the plot interactively with matplotlib.

    -x / --axes AXES
        Axis letters indicating which axes to plot and in what order.
        By default, corresponds to the number of ROWS plotted.  Default: 'xyz'.

    -n / --nslices NSLICES
        Number of slices to plot per dimension.  By default, corresponds to the
        number of COLUMNS plotted.  Slices will be approximately evenly spaced.
        Default: 7.

    -T / --transpose
        Transpose the grid, such that the collage will be SLICES x AXES
        instead of AXES x SLICES.

    --min / --max PROPORTION
        For all dimensions, set a range for plotting.  In a given dimension,
        the bottom --min (proportion) of voxels are not plotted, and the top
        (1 - --max) proprtion of voxels are not plotted.  By default this is
        not set.

    --minx / --maxx / --miny / --maxy / --minz / --maxz PROPORTION
        Set --min & --max for individual dimensions.  By default, all minimum
        values are 0.15 and maximum values are 0.85.

    -d / --dpi DPI
        DPI of the image.  Default is 300.

    --figx / --figy FIGSIZE
        Dimension of the output figure.  By default, this is set according
        to the number of rows and columns.

    -b / --background COLOR
        Background color.  Default: not set.

    -S / --separate
        When passed, the panels will be saved as individual images, rather than
        one large image.  This changes the behavior of -o / --output to be the
        folder where the images are saved.  The program will attempt to create
        this output folder if it does not exist.

        Note that the --figx / --figy arguments set the size of the overall
        grid of images, rather than individual panels.  As such, to increase
        or decrease the panel size, you would need to scale both these with the
        correct ratio.
    """

# https://stackoverflow.com/a/9028031/13386979
# for getting argmuents parsed in order
class CustomAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):

        if not 'ordered_args' in namespace:
            setattr(namespace, 'ordered_args', [])

        previous = namespace.ordered_args
        previous.append((self.dest, values))
        setattr(namespace, 'ordered_args', previous)

def parse_ordered_image_args(ordered_args):

    image_flags = {'anat', 'mask'}
    current_image = None
    images = []
    float_arguments = ['alpha', 'maskvalue', 'vmin', 'vmax']

    for key, value in ordered_args:

        if key in image_flags:

            if current_image is not None:
                images.append(current_image)

            current_image = {}
            current_image['type'] = key
            current_image['path'] = value

        else:

            if current_image is not None:

                value = float(value) if key in float_arguments else value
                current_image[key] = value

    if current_image is not None:
        images.append(current_image)

    return images

def parse():

    parser = argparse.ArgumentParser(add_help=False)

    # IMAGES

    # TYPES
    parser.add_argument('-A', '--anat', action=CustomAction)
    parser.add_argument('-M', '--mask', action=CustomAction)

    # IMAGE OPTIONS (ALL)
    parser.add_argument('-c', '--color', action=CustomAction)
    parser.add_argument('-a', '--alpha', action=CustomAction)

    # IMAGE OPTIONS (ANATOMY)
    parser.add_argument('-s', '--scalepanel', action=CustomAction, nargs=0)
    parser.add_argument('-z', '--dropzero', action=CustomAction, nargs=0)
    parser.add_argument('--vmin', action=CustomAction)
    parser.add_argument('--vmax', action=CustomAction)

    # IMAGE OPTIONS (MASK)
    parser.add_argument('-m', '--maskvalue', action=CustomAction)

    # GLOBALS
    parser.add_argument('-h', '--help', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-o', '--output')
    parser.add_argument('-P', '--plot', action='store_true')
    parser.add_argument('-x', '--axes', default='xyz')
    parser.add_argument('-n', '--nslices', default=7, type=int)
    parser.add_argument('-T', '--transpose', action='store_true')
    parser.add_argument('--min', default=None, type=float)
    parser.add_argument('--max', default=None, type=float)
    parser.add_argument('--minx', default=0.15, type=float)
    parser.add_argument('--maxx', default=0.85, type=float)
    parser.add_argument('--miny', default=0.15, type=float)
    parser.add_argument('--maxy', default=0.85, type=float)
    parser.add_argument('--minz', default=0.15, type=float)
    parser.add_argument('--maxz', default=0.85, type=float)
    parser.add_argument('-d', '--dpi', default=300, type=int)
    parser.add_argument('--figx', default=None, type=int)
    parser.add_argument('--figy', default=None, type=int)
    parser.add_argument('-b', '--background', default='black')
    parser.add_argument('-S', '--separate', action='store_true')

    args = parser.parse_args()

    return args

def parse_image_dict(d):

    try:
        imtype = d['type']
        del d['type']
    except KeyError:
        raise RuntimeError(f'Problem parsing CLI input: no "type" field provided for input: {d}')

    if imtype not in ['anat', 'mask']:
        raise RuntimeError(f'Problem parsing CLI input: unrecognized image type "{imtype}"')

    cls = Mask if imtype == 'mask' else Anatomy
    return cls(**d)

def main():

    args = parse()

    # detect empty call
    if not len(sys.argv) > 1:
        args.help = True

    if args.help:
        print(helptxt)
        return

    # no output specified
    if args.output is None and not args.plot:
        raise ValueError("Either output (-o) must be specified or the plot (-P) "
                         "option must be passed.  Run `nifti_overlay -h` for help.")

    if not args.plot:
        matplotlib.use('Agg')

    # parse some args
    figsize = args.figx, args.figy
    if figsize == (None, None):
        figsize = 'automatic'

    # setup overlay object
    overlay = NiftiOverlay(planes=args.axes,
                           nslices=args.nslices,
                           transpose=args.transpose,
                           min_all=args.min,
                           max_all=args.max,
                           minx=args.minx,
                           miny=args.miny,
                           minz=args.minz,
                           maxx=args.maxx,
                           maxy=args.maxy,
                           maxz=args.maxz,
                           background=args.background,
                           figsize=figsize,
                           dpi=args.dpi,
                           verbose=args.verbose)

    # we scale up the image a bit if saving separate panels
    if overlay.figsize == 'automatic' and args.separate:
        overlay.automatic_figsize_scale = 2.0

    # add images
    image_dicts = parse_ordered_image_args(args.ordered_args)
    for d in image_dicts:
        overlay.images.append(parse_image_dict(d))

    # run
    _ = overlay.plot()

    # handle outputs
    if args.plot:
        plt.show()

    if args.output:
        overlay.generate(args.output, separate=args.separate, rerun=False)

if __name__ == '__main__':
    main()
