#%%
import os
import soundfile as sf
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential,load_model
from keras.layers.core import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
import h5py
from pydub import AudioSegment
import os
import datetime
from scipy.io.wavfile import write
os.environ['KMP_DUPLICATE_LIB_OK']='True'

stereo_filepath = os.getcwd() + '/data/1.wav'
mono_filepath = os.getcwd() + "/data_monowavs/1.wav"
mp3_filepath = os.getcwd() + "/data_mp3/1.mp3"
input_filepath = os.getcwd() + "/data_input/1.wav"
output_filepath = os.getcwd() + "/output/1.wav"

#Feature extraction
def feature_extraction(x,fs):

	frame_length_s = 0.04 # window length in seconds
	frame_length = int(2**np.ceil(np.log2(fs*frame_length_s))) # 40ms window length in samples
	# set an overlap ratio of 50 %
	hop_length = frame_length//2

	# Compute STFT
	_,_,X = signal.stft(x, nfft=frame_length,noverlap=hop_length, fs=fs,nperseg=frame_length)
	number_frequencies, number_time_frames = X.shape
	X = np.abs(X)

	# Segmentation
	segment_length_s = 0.5 # segment length in seconds
	segment_length = int(2**np.ceil(np.log2(segment_length_s/frame_length_s))) # ~0.4s in samples

	# Trim the frames that can't be fitted into the segment size
	trimmed_X = X[:, :-(number_time_frames%segment_length)]

	# Segmentation (number of freqs x number of frames x number of segment x 1). The last dimension is 'channel'.
	features = trimmed_X.reshape((number_frequencies,segment_length,-1,1), order='F')
	# Transpose the feature to be in form (number of segment x number of freqs x number of frames x 1)
	return features.transpose((2,0,1,3))

#Converts stereo wav to mono wav
def file_process():
	sound = AudioSegment.from_wav(stereo_filepath)
	sound = sound.set_channels(1)
	sound.export(mono_filepath, format="wav")

#---Convert to mp3------
# AudioSegment.from_wav(mono_filepath).export(mp3_filepath, format="mp3")

# Extracts and saves extracted features to hdf5 file
def save_features(X_train,X_test,y_train,y_test):
	
	with h5py.File('data.hdf5', 'w') as f:
		f.create_dataset('X_train', data=X_train)
		f.create_dataset('X_test', data=X_test)
		f.create_dataset('y_train', data=y_train)
		f.create_dataset('y_test', data=y_test)

# Read extracted features from hdf5 file
def read_features():
	with h5py.File('data.hdf5', 'r') as f:
			X_train = f.get('X_train').value
			X_test = f.get('X_test').value
			y_train = f.get('y_train').value
			y_test = f.get('y_test').value
	return X_train, y_train, X_test, y_test

#---------CNN Model-------
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


#-------- TEST----------
#%% ------Get features------

# Read from the file
# X_train, y_train, X_test, y_test = read_features()

# Extract features manually
x1, fs = sf.read(mono_filepath)
groundtruth_features = feature_extraction(x1,fs)

x2, fs = sf.read(input_filepath)
input_features = feature_extraction(x2,fs)

X_train,X_test,y_train,y_test = train_test_split(input_features,groundtruth_features,test_size=0.2,random_state=0)
save_features(X_train,X_test,y_train,y_test)


#%% ------ Fit model
# model = get_model(X_train.shape)

# checkpoint = ModelCheckpoint("SRCNN_check.h5", monitor='val_loss', verbose=1, save_best_only=True,
#                                  save_weights_only=False, mode='min')
# callbacks_list = [checkpoint]
# model.fit(X_train, y_train, batch_size=32, validation_data=(X_test, y_test),
                #    shuffle=True, epochs=100)


# model.save('test-{date:%Y-%m-%d %H:%M:%S}.txt'.format( date=datetime.datetime.now() ))

#%% ----- PREDICT---------
model = load_model('mae_model.h5')
yhat = model.predict(X_test)
#%% ------RECONSTRUCT THE AUDIO--------

# Restore to the original shape
yhat = yhat.transpose((1,2,0,3))
yhat = yhat.reshape((yhat.shape[0],-1), order='F')
# Save output file
_, xrec = signal.istft(yhat, fs)
write(output_filepath,fs,xrec)

#%%
