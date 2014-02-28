soundengine
===========

Audio matching software originally written for CS4500.
Takes either two audio files, one audio file and a directory of audiofiles,
or two directories of audio files.
The program first does a fuzzy match by comparing the [Mel-Frequency Cepstral Coefficients] 
(http://en.wikipedia.org/wiki/Mel-frequency_cepstrum) of each audio file.
If the two files are considered likely to be a match, we compare a subset of
the [Fourier Transform](http://en.wikipedia.org/wiki/Fourier_transform) values.
