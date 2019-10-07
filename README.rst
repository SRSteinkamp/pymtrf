[![Build Status](https://travis-ci.org/SRSteinkamp/pymtrf.svg?branch=master)](https://travis-ci.org/SRSteinkamp/pymtrf)

pymtrf
======

pymtrf is a translation to Python 3.6 of the mTRF Toolbox (v.1.5) for MATLAB, which can be found at http://www.mee.tcd.ie/lalorlab/resources.html or at https://github.com/mickcrosse/mTRF-Toolbox.

original mTRF Toolbox Summary
-----------------------------

(copied from https://sourceforge.net/projects/aespa/?source=navbar )

mTRF Toolbox is a MATLAB toolbox that permits the fast computation of the linear stimulus-response mapping of any sensory system in the forward or backward direction. It is suitable for analysing EEG, MEG, ECoG and EMG data.

The forward model, or temporal response function (TRF), can be interpreted using conventional analysis techniques such as time-frequency and source analysis. The TRF can also be used to predict future responses of the system given a new stimulus signal. Similarly, the backward model can be used to reconstruct spectrotemporal stimulus information given new response data.

mTRF Toolbox facilitates the use of continuous stimuli in electrophysiological studies as opposed to time-locked averaging techniques which require discrete stimuli. This enables examination of how neural systems process more natural and ecologically valid stimuli such as speech, music, motion and contrast.

Support documentation: http://dx.doi.org/10.3389/fnhum.2016.00604

Dependencies
~~~~~~~~~~~~

pymtrf requires:

- Python (>=3.6)
- NumPy (>=1.14.0)
- SciPy (>=1.11.0)
- Pytest (5.0.1)

examples.ipynb requires additionally:
- seaborn
- matplotlib

Due to the use of the f'' format string the Python version ist set to 3.6, it is possible that in future releases this requirement will be removed. Furthermore, the requirements for NumPy and SciPy are (still) rather arbitrary, further testing will be required.

Installation
~~~~~~~~~~~~

Cone or download the repository and move (cd) into the pymtrf folder. Run :code:`pip install .`. This has so far only been test using pip 18.1. You can run the tests in the folder usinge :code:`python setup.py pytest`, this will require pytest. Another way is to install via pip and git+, however, this also downloads the example data, which is quite a lot (and proably not wanted).

Functions
=========

Inside the package
------------------

The functions in the Python version of mtrf are the same as in the MATLAB Toolbox, with similar use. Naming conventions have been adjusted for Python.

- lag_gen: Generate lagged timeseries
- mtrf_train: trains the linear model (backward and forward modeling)
- mtrf_predict: predicts and evaluates model
- mtrf_crossval: leave-one-out cross-validation function, does prediction and validation
- mtrf_multicrossval: similar to mtrf_crossval, allows multisensory responses
- mtrf_transform: transforms model weights, for better interpretability

Other
-----

- matlab_test_sets.m: Matlab script to recreate the 'test_files' folder (data created using MATLAB 2016b and mTRF Toolbox v1.5)
- mtrf_test_set.m: Legacy, used to validate pymtrf against mTRF Toolbox
- simulate_test_data.py: Used to simulate test cases for precision tests (Python and MATLAB instances of the Toolbox).

Usage
=====

See examples/examples.ipynb for more detailed use of the different functions.

Tips on Practical Use
=====================

See README.txt of the MATLAB Toolbox

- Ensure that the stimulus and response data have the same sample rate
  and number of samples.
- Downsample the data when conducting large-scale multivariate analyses
  to reduce running time, e.g., 128 Hz or 64 Hz.
- Normalise all data, e.g., between [-1,1] or [0,1] or z-score. This will
  stabalise regularisation across trials and enable a smaller parameter
  search.
- Enter the start and finish time lags in milliseconds. Enter positive
  lags for post-stimulus mapping and negative lags for pre-stimulus
  mapping. This is the same for both forward and backward mapping - the
  code will automatically reverse the lags for backward mapping.
- When using mtrf_predict, always enter the model in its original
  3-dimensional form, i.e., do not remove any singleton dimensions.
- When using mtrf_crossval, the trials do not have to be the same length,
  but using trials of the same length will optimise performance.
- When using mtrf_multicrossval, the trials in each of the three sensory
  conditions should correspond to the stimuli in STIM.


Example Data Sets
================

See README.txt of the MATLAB Toolbox

contrast_data.mat
This MATLAB file contains 3 variables. The first is a matrix consisting
of 120 seconds of 128-channel EEG data. The second is a vector consisting
of a normalised sequence of numbers that indicate the contrast of a
checkerboard that was presented during the EEG at a rate of 60 Hz. The
third is a scaler which represents the sample rate of the contrast signal
and EEG data (128 Hz). See Lalor et al. (2006) for further details.

coherentMotion_data.mat
This MATLAB file contains 3 variables. The first is a matrix consisting
of 200 seconds of 128-channel EEG data. The second is a vector consisting
of a normalised sequence of numbers that indicate the motion coherence of
a dot field that was presented during the EEG at a rate of 60 Hz. The
third is a scaler which represents the sample rate of the motion signal
and EEG data (128 Hz). See Gonçalves et al. (2014) for further details.

speech_data.mat
This MATLAB file contains 4 variables. The first is a matrix consisting
of 120 seconds of 128-channel EEG data. The second is a matrix consisting
of a speech spectrogram. This was calculated by band-pass filtering the
speech signal into 128 logarithmically-spaced frequency bands between 100
and 4000 Hz and taking the Hilbert transform at each frequency band. The
spectrogram was then downsampled to 16 frequency bands by averaging
across every 8 neighbouring frequency bands. The third variable is the
broadband envelope, obtained by taking the mean across the 16 narrowband
envelopes. The fourth variable is a scaler which represents the sample
rate of the envelope, spectrogram and EEG data (128 Hz). See Lalor &
Foxe (2010) for further details.


References
==========

- Lalor EC, Pearlmutter BA, Reilly RB, McDarby G, Foxe JJ (2006) The
  VESPA: a method for the rapid estimation of a visual evoked potential.
  NeuroImage 32:1549-1561. https://doi.org/10.1016/j.neuroimage.2006.05.054
- Gonçalves NR, Whelan R, Foxe JJ, Lalor EC (2014) Towards obtaining
  spatiotemporally precise responses to continuous sensory stimuli in
  humans: a general linear modeling approach to EEG. NeuroImage 97(2014):196-205.
  https://doi.org/10.1016/j.neuroimage.2014.04.012
- Lalor, EC, & Foxe, JJ (2010) Neural responses to uninterrupted natural
  speech can be extracted with precise temporal resolution. Eur J Neurosci
  31(1):189-193. https://doi.org/10.1111/j.1460-9568.2009.07055.x
- Crosse MC, Di Liberto GM, Bednar A, Lalor EC (2015) The multivariate
  temporal response function (mTRF) toolbox: a MATLAB toolbox for relating
  neural signals to continuous stimuli. Front Hum Neurosci 10:604.
  https://dx.doi.org/10.3389%2Ffnhum.2016.00604
- Haufe S, Meinecke F, Gorgen K, Dahne S, Haynes JD, Blankertz B,
  Bießmann F (2014) On the interpretation of weight vectors of
  linear models in multivariate neuroimaging. NeuroImage 87:96-110.
  https://doi.org/10.1016/j.neuroimage.2013.10.067
- Crosse MC, Butler JS, Lalor EC (2015) Congruent visual speech
  enhances cortical entrainment to continuous auditory speech in
  noise-free conditions. J Neurosci 35(42):14195-14204.
  https://doi.org/10.1523/JNEUROSCI.1829-15.2015

TODO
====

- Extensive documentation
- More tests
- Tutorial to the method
- mtrf_predict, allow prediction only (skipping evaluation step)

Wishlist
========

- mtrf_class following scikit-learn API
- mne-python workflow (need data set...)
