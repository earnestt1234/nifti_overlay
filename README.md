# Nifti Overlay

Nifti Overlay is a Python program for creating visualizations of 3D volumetric neuorimaging data.  It specifically creates tiled images of one or more images overlaid on top of one of another.

Here is an example output image, showing a registered PET image on top of a T1-weighted MRI:

![PET image on T1 MRI](/docs/gallery/pet_on_t1.png) 

## Why use this package?

[I](https://github.com/earnestt1234/) wrote this package to help with quality control for neuroimaging preprocessing pipelines.  Most of the images I worked with were stored remotely, and it was fairly infeasible to look at them interactively.  I wrote (a prototype) of this package in order to generate quick snapshots of various steps of the pipeline.  For example, I would check the overlap of an automatically generated brain mask against the brain.  Or look at the alignment of two registered images.

So, if you want to quickly generate a picture that shows a lot of the brain, maybe this will be helpful.

## Installation

Nifti Overlay can be installed from PyPI:

```bash
pip install nifti_overlay
```

See the [releases page](https://github.com/earnestt1234/nifti_overlay/releases) for information about the most current release.

## Documentation

Check out the documentation: [https://nifti-overlay.readthedocs.io/en/latest/](https://nifti-overlay.readthedocs.io/en/latest/).

## License

Open source under [MIT](https://github.com/earnestt1234/nifti_overlay/blob/main/LICENSE).