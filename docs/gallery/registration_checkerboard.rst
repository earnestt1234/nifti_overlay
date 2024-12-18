
Registration: Checkerboard
---------------------------

.. image:: registration_checkerboard.png
  :width: 800
  :alt: Checkerboarding of a T1 image and the template.

CLI
++++++

.. code-block:: bash

    nifti_overlay -C registered_t1.nii.gz mni.nii.gz

Python
++++++

.. code-block:: python

    from nifti_overlay import NiftiOverlay

    overlay = NiftiOverlay()
    overlay.add_checkerboard(['registered_t1.nii.gz', 'mni.nii.gz'])
    overlay.plot()