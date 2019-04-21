#%%
from utils import save_features, get_features, read_features, feature_extraction
from model import get_model
from data_prepare import stereo_to_mono, compress
import os
from sklearn.model_selection import train_test_split
from scipy.io.wavfile import write
import soundfile as sf
from keras.models import load_model
from scipy import signal
#%% Parameters
input_folder = os.path.join(os.getcwd(), 'data_input')
mono_folder = os.path.join(os.getcwd(), 'data_mono')
stereo_folder = os.path.join(os.getcwd(), 'data')
input_filename = os.path.join(input_folder, 'other.wav')
batch_size =16
nb_epoch = 100
optimizer = 'adam'
loss = 'mae'
metrics = 'mae'
n_layers = 3

#%%
def main():
    if not os.path.exists('myData.h5py'):
        # prepare the data
        stereo_to_mono(stereo_folder, mono_folder)
        compress(mono_folder, input_folder)

        # extract features
        gt_features, gt_phase = get_features(mono_folder)
        input_features, input_phase = get_features(input_folder)
        X_train,X_test,y_train,y_test = \
            train_test_split(input_features, gt_features, test_size=0.2)
        print(X_train.shape)
        print(X_test.shape)
        print(len(gt_phase))
        print(len(input_phase))
    
        # save features
        save_features('myData.h5py', X_train, X_test, y_train, y_test)
    else:
        X_train, y_train, X_test, y_test = read_features('myData.h5py')
    
    # get the model and train
    model = get_model(y_train.shape)
    model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test), \
              shuffle=True, epochs=nb_epoch)   
    model.save('model-{}batch-{}epochs.h5'.format(batch_size,  nb_epoch))
    
    
    # predict and generate output files
    model = load_model('model-16batch-1epochs.h5')
    y, fs = sf.read(input_filename)
    phaseInfo, feat = feature_extraction(y,fs)
    yhat = model.predict(feat)
    
    # reconstruct the audio
    yrec = yhat.transpose((1,2,0,3))
    yrec = yrec.reshape((yrec.shape[0],-1), order='F')
    # yrec = yrec + phaseInfo
    _, xrec = signal.istft(yrec, fs)
    write("output.wav",fs,xrec)
    print('Output without phase saved.')
    yrec = yrec + phaseInfo
    # yrec = np.vstack((yrec,np.flipud(yrec)))
    _, xrec = signal.istft(yrec, fs)
    write("output_with_phase.wav",fs,xrec)
    print('Output with phase saved.')
    
    
if __name__ == "__main__":
    main()

