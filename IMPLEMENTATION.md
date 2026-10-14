**DATA ACQUISITION & PREPROCESSING**

Motorcycle sound samples are extremely messy as sounds from the motorcycle (engine pistons, chain, exhaust, bearings, wheels), environmental noise and air flow are obtained all together.

Iterative cycles of rigorous Band Pass Filtering was executed to isolate useful sound samples; 
Useful sound sample for, 
1. healthy: motorcycle engine sound
2. chain fault: engine sound with excess chain component with different frequency, (higher than normal)
3. exhaust leak: engine sound with a distinct high amplitude component
4. engine tappet noise: same frequency with engine sound but different intervals.

Hardware used for DAQ: INMP441, an i2s based MEMS Microphone, data obtained in 16 kHz sampling rate and an output of 24 bit PCM audio data

Sound signals are in the form of 512 frames
Windowing: Hamming Window
FFT is performed to convert the representation from time domain to frequency domain

**MFCC**

Mel's Frequency Cepstral Coefficients

DSP algorithm with a series of Mathematical Processes

1. Mel Filter Bank is used to convert frames into triangular bands called "Mel's Bands", They respresent spectral energy
2. Logarithm is used to compress the energy bands
3. Discrete Cosine Transform is used to convert the energy bands into cepstral coefficients

**Feature Vectors**

The coefficients for an entire 5 second audio is calculated and statistically aggregated in terms of mean, median, standard deviation, spectral centroid & RMS
They are represented as feature vectors.

**Reference Vectors**

Pre-determined feature vectors to use as reference with feature vectors of the test samples
Reference Vectors are stored in the Flash Memory.

**Vector Similarity Analysis: Detection and Classification**

Vector Similarity Analysis between feature vectors and reference vectors are used to detect faults

Cosine Similarity and Pearson's Coefficients of test sample feature vectors with reference vectors of healthy, chain, tappet and exhaust are calculated.
Fault detection condition: Cosine similarity > 0.9 || Pearson's Coefficient > 0.85

Classification is determined by closeness to each of the reference vectors
