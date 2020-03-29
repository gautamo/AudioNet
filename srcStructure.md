# Piano-Audio-Classifier README FOR FILES IN SRC

Directories:

data_root2: contains the train/validation/test set data for a sample of the total data. All 35K+ classes are represented. Each class has 11 images classified into that class.

msmd: git repo with helper functions to manage msmd_aug_v1-1_no-audio dataset.

msmd_aug_v1-1_no-audio: folder containing 14 performances of total 601 performances with MIDI matrix and sheet music.

CODE/models: AudioNet neural network saved weights are stored here.

Files:

convFilter.py: contains helper functions to manipulate sprectrogram images into slices and can convert midi matrix into textual music notes.

CODE/audioNet.py: contains AudioNet neural network model and training/evaluation functions.

CODE/audio_to_spectrogram.py: contains code to convert audio to a spectrogram.

CODE/automationTests.ipynb: testing framework used to evaluate different AudioNet architectures and hyperparameters. Contains 25 tests.

CODE/BaseLineClassifier.ipynb: baseline Perceptron Classifier used to compare performance to AudioNet.

CODE/Conversion & Filter Examples.ipynb: notebook used to explain functions provided in convFilter.py.

CODE/convFilter.py: contains helper functions to manipulate sprectrogram images into slices and can convert midi matrix into textual music notes.

CODE/dataSorting.py: contains functions to preprocess data from spectrograms and splits spectrograms into train/test/validation sets. Also used to count data pieces saved in data_root.

