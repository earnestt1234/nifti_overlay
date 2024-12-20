
Usage: Python
=============

This page will cover how to use Nifti Overlay from within a Python session.  The workflow is very similar to that of the CLI, so reading that usage page (:ref:`usage_cli`) might also be helpful.

Basic Usage
-----------

Start by importing the :class:`nifti_overlay.core.NiftiOverlay` class.

.. code-block:: python

    from nifti_overlay import NiftiOverlay

For every visualization, create an instance of the NiftiOverlay class.

.. code-block:: python

    overlay = NiftiOverlay()

You can then add layers to be plotted, using one of the ``.add_...()`` methods:

- :func:`~nifti_overlay.core.NiftiOverlay.add_anat` Plot the raw intensities of an image.
- :func:`~nifti_overlay.core.NiftiOverlay.add_mask` Plot only a single value within an image.
- :func:`~nifti_overlay.core.NiftiOverlay.add_edges` Plot the countours of an image.
- :func:`~nifti_overlay.core.NiftiOverlay.add_checkerboard` Plot 2 (or more) images interleaved with each other in a checkerboard pattern.

For example:

.. code-block:: python

    overlay.add_anat('t1.nii.gz')
    overlay.add_mask('lesion_mask.nii.gz')

Images are parsed into objects (see :py:mod:`nifti_overlay.image` and :py:mod:`nifti_overlay.multiimage`), and stored in the ``images`` attribute of the NiftiOverlay instance.

Creating the plot
-----------------

Once a NiftiOverlay class is instantiated and the desired images have been added, there are two methods which can be used for generating the visualization:

- :func:`nifti_overlay.core.NiftiOverlay.plot` will run the visualization.  The plot will then be displayed if working in an environment where matplotlib can show plots.
- :func:`nifti_overlay.core.NiftiOverlay.generate` will also run the visualization, but will save the visualization as an image file.

Either of these methhods should be called at the end of the script, after all desired images have been added.  When plotting, a new matplotlib Figure and several Axes will be created (these are accessible by the ``fig`` and ``axes`` attributes of the NiftiOverlay instance).

Options
-------

There are two places where options can be specified to customize the visualization:

- During the instantiation of :class:`nifti_overlay.core.NiftiOverlay`.  These are global options which affect the entire visualization for all images plotted.  Global options can change the number of slices/axes plotted, the DPI, the figure size, and others.
- When calling one of the ``add_...()`` methods.  These are options which are specific to one image/layer being plotted.  This is where the color/opacity of the layer can be set, as well as other layer-specific configurations.

As an example:

.. code-block:: python

    # set global options when initializing the overlay
    overlay = NiftiOverlay(
        nslices=5,          # plot 5 slices for each image axis
        planes='zyx',       # select which axes are plotted (and their order)
        miny=0.4,           # set the range for over which slices are slected in the y-direction
        maxy=0.6,
        dpi=300             # set the DPI
    )

    # now set specific arguments for each image/layer being plotted
    overlay.add_anat(
        src='t1.nii.gz',    # path to the image
        color='viridis',    # colormap to use
    )

    overlay.add_mask(
        src='mask.nii.gz',  # path to the image
        color='cyan',       # color for this image; only affects this image
        alpha=0.5,          # opacity; only affects this image 
    )

See :class:`nifti_overlay.core.NiftiOverlay` for a complete list of global and layer-specific arguments.  Also, see the gallery for more Python examples (:ref:`gallery`).