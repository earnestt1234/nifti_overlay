
.. _usage_cli:

Usage: Command Line
===================

When installed (see :ref:`installation`), Nifti Overlay can be accessed from the
command line, using the command ``nifti_overlay``.  See the docstring for the
command line interface (CLI) like so:

.. code-block:: bash

    nifti_overlay -h

The source code for the CLI can be viewed `here <https://github.com/earnestt1234/nifti_overlay/blob/main/nifti_overlay/__main__.py>`__.

Basic Usage
-----------

There are three primary flavors of arguments that can be supplied to ``nifti_overlay``:

1. Paths to images you want to visualize
2. Image-specific options (which modify the plotting of images they are bound to)
3. Global options (which modify the plot for everything being plotted)

You must provide at least one path to an image to be plotted.  There are
a few different modes for plotting images:

- **Anatomy** (``-A``, ``--anat``): The most basic plot, which plots the raw values from an image.  If you want to view a continuously valued image (from a structural MRI, PET, CT, or whatever), use this option to add images.
- **Mask** (``-M``, ``--mask``): Plots a binary mask or a single label from a segmentation.  A single color is used to show the spatial distribution of a label.
- **Edges** (``-E``, ``--edges``): For each slice, an automated edge detection is applied.  The extracted edges are plotted.
- **Checkerboard** (``-C``, ``--checker``): Use 2 (or more) images, created a checkerboard pattern where images are interleaved with each other.  The typical use case for this is checking registration alignment.

With this information, the most basic call is the specifiy a single image to plot:

.. code-block:: bash

    nifti_overlay -A t1.nii.gz

There is no limit on how many images you can plot; adding more images will overlay
one on top of the other (with the last specified one appearing "on top").

.. code-block:: bash

    nifti_overlay -A t1.nii.gz -M brain_mask.nii.gz

Each mode of plotting has specific options which will alter the plotting of a single image.
**Image-specific arguments are applied to the preceeding image path**.

.. code-block:: bash

    nifti_overlay -A t1.nii.gz --alpha 1.0 -M brain_mask.nii.gz --alpha 0.7

In theabove example, unique opacity values (``--alpha``) are specified for the anatomy (1.0) and the mask (0.7).

Global arguments, on the other hand, can be specified wherever in the call.  For example, these calls (which increase the number of slices plotted) will produce the same output:

.. code-block:: bash

    nifti_overlay -A t1.nii.gz -M brain_mask.nii.gz --nslices 12
    nifti_overlay -A t1.nii.gz --nslices 12 -M brain_mask.nii.gz

See the help docstring for a full list of image-specific and global options.

Outputs
-------

By default, any created visualizations are shown in an `interactive matplotlib window <https://matplotlib.org/stable/users/explain/figure/interactive.html>`__.  After closing this display, the program exits and the image is not saved (although the interactive viewer can be used to save the image).

To save the image to file, specify the ``-o``/``--output`` argument with a path
to an image (see `here <https://stackoverflow.com/a/7608273/13386979>`__ for a discussion of acceptable image formats).

.. code-block:: bash

    nifti_overlay -A t1.nii.gz -M brain_mask.nii.gz --nslices 12 -o my_image.png

When writing to a file, the plot is not shown interactively, unless ``-P``/``--plot`` is included.