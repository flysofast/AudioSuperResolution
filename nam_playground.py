#%%
import os
import soundfile as sf
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

#%% ----------Feature extraction-------------
def feature_extraction(x,fs):
    duration = len(x)/float(fs)
    frame_length_s = 0.04 # window length in seconds
    frame_length = int(2**np.ceil(np.log2(fs*frame_length_s))) # 40ms window length in samples
    # set an overlap ratio of 50 %
    hop_length = frame_length//2

    # Compute STFT
    _,_,X = signal.stft(x, nfft=frame_length,noverlap=hop_length, fs=fs,nperseg=frame_length)
    number_frequencies, number_time_frames = X.shape

    #%% Segmentation
    segment_length_s = 0.5 # segment length in seconds
    segment_length = int(2**np.ceil(np.log2(segment_length_s/frame_length_s))) # ~0.4s in samples

    # Trim the frames that can't be fitted into the segment size
    trimmed_X = X[:, :-(number_time_frames%segment_length)]

    # Segmentation (number of freqs x number of frames x batch number)
    features = trimmed_X.reshape((number_frequencies,segment_length,-1), order='F')
    # Transpose the feature to be in form (batch number x number of freqs x number of frames)
    return features.transpose((2,0,1))

    # #%% --------Spectrogram--------------
    # # The whole file spectrogram------------------
    # to_be_plotted = trimmed_X
    # freq_scale = np.linspace(0, fs / 2, to_be_plotted.shape[0])
    # timeframe_scale = np.linspace(0, duration, to_be_plotted.shape[1])

    # # plot spectrogram (amplitude only)
    # W = np.abs(to_be_plotted)
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.pcolormesh(timeframe_scale, freq_scale, np.log(W+0.00001))
    # plt.xlabel('Time (s)')
    # plt.ylabel('Frequency (Hz)')
    # plt.title('Entire file Spectrogram (log scale)')
    # plt.tight_layout()

    # # First batch spectrogram--------------
    # freq_scale = np.linspace(0, fs / 2, number_frequencies)
    # timeframe_scale = np.linspace(0, segment_length*frame_length_s, segment_length)
    # # plot spectrogram (amplitude only)
    # W = np.abs(features[50])
    # plt.subplot(2,1,2)
    # plt.pcolormesh(timeframe_scale, freq_scale, np.log(W+0.00001))
    # plt.xlabel('Time (s)')
    # plt.ylabel('Frequency (Hz)')
    # plt.title('First batch spectrogram (log scale)')
    # plt.tight_layout()
    # plt.show()

#%% -------Convert stereo wav to mono wav------------
stereo_filepath = os.getcwd() + '/data/1.wav'
mono_filepath = os.getcwd() + "/data_monowavs/1.wav"
mp3_filepath = os.getcwd() + "/data_mp3/1.mp3"
from pydub import AudioSegment
sound = AudioSegment.from_wav(stereo_filepath)
sound = sound.set_channels(1)
sound.export("data_monowavs/1.wav", format="wav")

#%%---Convert to mp3------
AudioSegment.from_wav(mono_filepath).export(mp3_filepath, format="mp3")
#%%---------Test-----------
x, fs = sf.read(mono_filepath)
features = feature_extraction(x,fs)

#%%---------CNN Model-------
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
def get_model():
    N = 32 # Number of feature maps 
    w, h = 3, 3 # Conv. window size
    input_shape = features.shape[1:] # (number of freqs x number of frames in a segment)

    model = Sequential()
    model.add(Conv2D(N, (w, h),
            input_shape=input_shape,
            activation = "relu",
            padding = "same"))
    # model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Conv2D(N, (w, h),
            activation = "relu",
            padding = "same"))
            
    model.add(Conv2D(N, (w, h),
            activation = "relu",
            padding = "same"))

    model.summary()
    return model;

