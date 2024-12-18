
Registration: Edges
-------------------

.. image:: registration_edges.png
  :width: 800
  :alt: Edges of MNI template ovelaid on registered T1 image.

CLI
++++++

.. code-block:: bash

    nifti_overlay -A registered_t1.nii.gz -E mni.nii.gz --sigma 3

Python
++++++

.. code-block:: python

    from nifti_overlay import NiftiOverlay

    overlay = NiftiOverlay()
    overlay.add_anat('registered_t1.nii.gz')
    overlay.add_edges('mni.nii.gz', sigma=3)
    overlay.plot()