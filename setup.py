from os import path

from setuptools import setup

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# read version
with open(path.join(this_directory, 'nifti_overlay', '__version__.py'), encoding='utf-8') as f:
    version = f.read().split('=')[1].strip('\'"\n')

dependencies = [
    "matplotlib>=3.2",
    "nibabel>=3.2",
    "numpy>=1.2",
    "scikit-image>=0.24.0"
]

setup(name='nifti_overlay',
      version=version,
      python_requires='>=3.7',
      description='A program for creating tiled images of volumetric neuorimaging data.',
      url='https://github.com/earnestt1234/nifti_overlay',
      author='Tom Earnest',
      author_email='earnestt1234@gmail.com',
      license='MIT',
      packages=['nifti_overlay'],
      install_requires=dependencies,
      long_description=long_description,
      long_description_content_type='text/markdown',
      entry_points = {
        'console_scripts': ['nifti_overlay=nifti_overlay.__main__:main'],
        })
