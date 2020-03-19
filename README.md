# Acivity classification

*Technologies:* Python, NumPy, Plotly, PyTorch, PyWavelets, Tensorboard, H5py, Tqdm, PIL. <br>
*Topics:* classification, signal processing, time series. <br>

## Description

<p float="left">
  <img src="readme_files/spectrograms.png?raw=true" width="400" />
</p>

In this project, we use three different approaches to classify temporal signals according to the associated activity. Our input data consists of several thousand short snippets of measurements obtained from nine sensors (such as acceleration and gyroscope) while people performed six different activities (such as walking or sitting). In our first approach, we train a simple feed-forward network using the raw temporal signals and associated labels. In our second approach, we compute spectrograms by applying a Gabor transform to the temporal signals, and train a CNN to classify the spectrograms. In our third approach, we compute scaleograms by using a continuous wavelet transform, and train a CNN to classify the scaleograms.

You can find more details in the <a href="https://1drv.ms/b/s!AiCY1Uw6PbEfheUz1u-obevm2AsltA?e=U0Ls2N">report</a> for this project.

This was my final project for the Computational Methods for Data Analysis class (AMATH 582) at the University of Washington, which I completed as part 
of my masters in Applied Mathematics.

## Running

To run this project:

```sh
conda env create -f environment.yml
conda activate activity-classification
python main.py
```