# A General Solution to Hot Pixels for Event Camera

<p align="center" width="100%">
    <img width="50%" src="fig/hot_pixel_package_logo2.png">
</p>

# Objective

A pixel-wise approach to remove any short and long lasting hot pixels from any event stream without knowing the true number of the hot pixels, using just a single parameter.

# Installation

```
conda create --name hot_pixel_filter python=3.9
conda activate hot_pixel_filter
python3 -m pip install -e .
pip install torch
pip install tqdm
pip install plotly
pip install scikit-image
pip install loris
pip install PyYAML
pip install opencv-python
pip install scikit-learn
pip install hdbscan
pip install astroquery
pip install pillow
python3 -m pip install astropy requests astrometry
python3 -m pip install scikit-image matplotlib-label-lines ipywidgets
conda install -c conda-forge pydensecrf

sudo apt install -y build-essentials
```


# Run
