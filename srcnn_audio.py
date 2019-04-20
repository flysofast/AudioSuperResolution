# -*- coding: utf-8 -*-
# author: lehainam
"""# **Features extraction and train the model**

**Function definitions:**
"""

#%%
import os
import soundfile as sf
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
import h5py
from pydub import AudioSegment
import os
import datetime

stereo_filepath = os.getcwd() + '/data/1.wav'
mono_filepath = os.getcwd() +"/data_monowavs/1.wav"
mp3_filepath = os.getcwd() + "/data_mp3/1.mp3"
input_filepath = os.getcwd() + "/data_input/1.wav"

input_folder = os.path.join(os.getcwd(), 'data_input')
mono_folder = os.path.join(os.getcwd(), 'data_monowavs')
stereo_folder = os.path.join(os.getcwd(), 'data')
#Feature extraction
def feature_extraction(x,fs):

	frame_length_s = 0.04 # window length in seconds
	frame_length = int(fs*frame_length_s) # 40ms window length in samples
	# set an overlap ratio of 50 %
	hop_length = frame_length//2

	# Compute STFT
	_,_,X = signal.stft(x, noverlap=hop_length, fs=fs,nperseg=frame_length)
	number_frequencies, number_time_frames = X.shape
	phaseInfo = np.angle(X)
	X = np.abs(X)

	# Segmentation
	sample_length_s = 0.8 # segment length in seconds
	sample_length = int(sample_length_s/frame_length_s) # ~1s in samples

	# Trim the frames that can't be fitted into the segment size
	trimmed_X = X[:, :-(number_time_frames%sample_length)]
	trimmed_phaseInfo = phaseInfo[:, :-(number_time_frames%sample_length)]

	# Segmentation (number of freqs x number of frames x number of segment x 1). The last dimension is 'channel'.
	features = trimmed_X.reshape((number_frequencies,sample_length,-1,1), order='F')
	# Transpose the feature to be in form (number of segment x number of freqs x number of frames x 1)
	return trimmed_phaseInfo,features.transpose((2,0,1,3))

#Converts stereo wav to mono wav
def file_process(file_path):
	_, filename = os.path.split(file_path)
	sound = AudioSegment.from_wav(file_path)
	sound = sound.set_channels(1)
	sound.export(os.path.join(mono_folder,'mono_'+filename), format="wav")


#%%---Convert to mp3------
# AudioSegment.from_wav(mono_filepath).export(mp3_filepath, format="mp3")
#%%---------Test extract features-----------


def save_features(X_train,X_test,y_train,y_test):

	with h5py.File(features_filename, 'w') as f:
		f.create_dataset('X_train', data=X_train)
		f.create_dataset('X_test', data=X_test)
		f.create_dataset('y_train', data=y_train)
		f.create_dataset('y_test', data=y_test)

	return X_train,X_test,y_train,y_test

#%% -----Read data-----------

def read_features():
	with h5py.File(features_filename, 'r') as f:
			X_train = f.get('X_train').value
			X_test = f.get('X_test').value
			y_train = f.get('y_train').value
			y_test = f.get('y_test').value
	return X_train, y_train, X_test, y_test


#%%---------CNN Model-------

def get_model(features_shape):
	input_shape = (features_shape[1],features_shape[2], 1)# (number of freqs x number of frames in a segment x number of channels)
	model = Sequential()
	model.add(Conv2D(32, (5, 5),
			input_shape=input_shape,
			activation = "relu",
			padding = "same"))
	# model.add(MaxPooling2D(pool_size=(4, 4)))
	model.add(Conv2D(64, (5, 5),
			activation = "relu",
			padding = "same"))
			
	model.add(Conv2D(1, (10, 10),
			activation = "relu",
			padding = "same"))

	adam = Adam(lr=0.0003)
	model.compile(optimizer=adam, loss='mean_absolute_error', metrics=['mean_absolute_error'])
	model.summary()
	return model

"""**Extract features**"""

# Read from file
#  X_train, y_train, X_test, y_test = read_features()

# Extract features manually
x1, fs = sf.read(groundtruth_filename)
_,groundtruth_features = feature_extraction(x1,fs)

x2, fs = sf.read(input_filename)
_ ,input_features = feature_extraction(x2,fs)

X_train,X_test,y_train,y_test = train_test_split(input_features,groundtruth_features,test_size=0.2)
save_features(X_train,X_test,y_train,y_test)

"""**Train the model:**"""

model = get_model(y_train.shape)

# checkpoint = ModelCheckpoint("SRCNN_check.h5", monitor='val_loss', verbose=1, save_best_only=True,
#                                  save_weights_only=False, mode='min')
# callbacks_list = [checkpoint]
model.fit(X_train, y_train, batch_size=16, validation_data=(X_test, y_test),
                   shuffle=True, epochs=100)
optimizer = 'adam'
loss = 'mae'
metrics = 'mae'
n_layers = 3

model.save('test-{date:%Y-%m-%d %H:%M:%S}.h5'.format( date=datetime.datetime.now() ))

"""# **Generate the output file**

---
"""

from scipy.io.wavfile import write
#%% ----- PREDICT---------

# model = load_model('mae_model.h5')

y, fs = sf.read(input_filename)
phaseInfo,feat = feature_extraction(y,fs)
yhat = model.predict(feat)
#%% ------RECONSTRUCT THE AUDIO--------

# Restore to the original shape
yrec = yhat.transpose((1,2,0,3))
yrec = yrec.reshape((yrec.shape[0],-1), order='F')
# yrec = yrec + phaseInfo
# yrec = np.vstack((yrec,np.flipud(yrec)))
# Save output file
_, xrec = signal.istft(yrec, fs)
write("output.wav",fs,xrec)
print('Output without phase saved.')

yrec = yrec + phaseInfo
# yrec = np.vstack((yrec,np.flipud(yrec)))
# Save output file
_, xrec = signal.istft(yrec, fs)
write("output_with_phase.wav",fs,xrec)
print('Output with phase saved.')
#%%

