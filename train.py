#%%
from utils import save_features, get_features, read_features, feature_extraction, reconstruct
from model import get_model
from data_prepare import stereo_to_mono, compress
import os
from sklearn.model_selection import train_test_split
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
model_name = 'model-{}batch-{}epochs.h5'.format(batch_size,  nb_epoch)
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
        print("X_train shape: ", X_train.shape)
        print("X_test shape: ", X_test.shape)
        # print(len(gt_phase))
        # print(len(input_phase))
    
        # save features
        save_features('myData.h5py', X_train, X_test, y_train, y_test)
    else:
        X_train, y_train, X_test, y_test = read_features('myData.h5py')
    
    # get the model and train
    if not os.path.exists(model_name):
        model = get_model(y_train.shape)
        model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test), \
                  shuffle=True, epochs=nb_epoch)
        model.save(model_name)
    
    
    # predict and generate output files
    model = load_model(model_name)
    y, fs = sf.read(input_filename)
    reconstruct(y,fs,model)
    
    
if __name__ == "__main__":
    main()

