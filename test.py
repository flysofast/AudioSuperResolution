# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 13:05:28 2019

@author: Hung
"""

#%%
#IMPORT LIBRARIES
from utils import save_features, get_features, read_features, feature_extraction, reconstruct, split
from model import get_model
from data_prepare import stereo_to_mono, compress
import os
from sklearn.model_selection import train_test_split
import soundfile as sf
from keras.models import load_model
from scipy import signal
from keras.callbacks import ModelCheckpoint,EarlyStopping
import datetime
import librosa
import matplotlib.pyplot as plt
import scipy
import numpy as np

#%%
# PREPARE DATA
#stereo_to_mono('rawdata','groundtruth')
#compress('groundtruth','training_samples')
test_data,test_fs = sf.read('training_samples/bass.wav')
orig_data,orig_fs = sf.read('groundtruth/bass.wav')

#%%
# LOAD MODEL
model = load_model('SRCNN_2019-05-01 14_42_26_best.h5')

#%%
# Reconstruct
predict = reconstruct(test_data,test_fs,model)

#%% 
# SPECTROGRAM ANALYSIS
output_data,output_fs = sf.read('output_with_phase.wav')

# Plot spectrogram of original data
plt.figure(0)
orig_f,orig_t,orig_spec = scipy.signal.stft(orig_data,orig_fs)
plt.pcolormesh(orig_t, orig_f, 20 * np.log10(np.abs(orig_spec) + 0.0001))
plt.title('Spectrogram of original high-quality data')
plt.xlabel('Time (s)')
plt.ylabel('Freq (Hz)')

# Plot spectrogram of test data (downsampled)
plt.figure(1)
test_f, test_t, test_spec = scipy.signal.stft(test_data,test_fs)
plt.pcolormesh(test_t, test_f, 20 * np.log10(np.abs(test_spec) + 0.0001))
plt.title('Spectrogram of test data (downsampled)')
plt.xlabel('Time (s)')
plt.ylabel('Freq (Hz)')


# Plot spectrogram of created high-quality data
plt.figure(2)
predict_f, predict_t, predict_spec = scipy.signal.stft(output_data,output_fs)
plt.pcolormesh(predict_t, predict_f, 20 * np.log10(np.abs(predict_spec) + 0.0001))
plt.title('Spectrogram of output')
plt.xlabel('Time (s)')
plt.ylabel('Freq (Hz)')

# Calculate mse
#orig_vs_test = np.mean((orig_data[:len(output_data)]-test_data[:len(output_data)])**2)
#orig_vs_output = np.mean((orig_data[:len(output_data)]-output_data)**2)

# Calculate mae
orig_vs_test = np.mean(np.abs((orig_data[:len(output_data)]-test_data[:len(output_data)])))
orig_vs_output = np.mean(np.abs((orig_data[:len(output_data)]-output_data)))
print('MAE of original vs downsampled files: ',orig_vs_test)
print('MAE of original vs output files: ',orig_vs_output)


