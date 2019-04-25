#%%
from utils import save_features, get_features, read_features, feature_extraction, reconstruct, split, unison_shuffled_copies
from model import get_model
from data_prepare import stereo_to_mono, compress
import os
from sklearn.model_selection import train_test_split
import soundfile as sf
from keras.models import load_model
from scipy import signal
from keras.callbacks import ModelCheckpoint,EarlyStopping
import datetime
#%% Parameters
input_folder = os.path.join(os.getcwd(), 'training_samples')
groundtruth_folder = os.path.join(os.getcwd(), 'groundtruth')
stereo_folder = os.path.join(os.getcwd(), 'rawdata')
eval_stereo_folder = os.path.join(os.getcwd(), 'eval_rawdata')
eval_groundtruth_folder = os.path.join(os.getcwd(), 'eval_groundtruth')
eval_input_folder = os.path.join(os.getcwd(), 'eval_training')
input_filename = os.path.join(input_folder, 'other.wav')
batch_size =16
nb_epoch = 100

# model_name = 'model-{}batch-{}epochs.h5'.format(batch_size,  nb_epoch)
#%%
def main():
    if not os.path.exists('myData.h5py'):
        # prepare the data
        stereo_to_mono(stereo_folder, groundtruth_folder)
        compress(groundtruth_folder, input_folder)
        stereo_to_mono(eval_stereo_folder, eval_groundtruth_folder)
        compress(eval_groundtruth_folder, eval_input_folder)
        
        # extract features
        gt_features, _ = get_features(groundtruth_folder)
        input_features, _ = get_features(input_folder)
        eval_gt_features, _ = get_features(eval_groundtruth_folder)
        eval_input_features, _ = get_features(eval_input_folder)
        
        # shuffle features
        gt_features, input_features = unison_shuffled_copies(gt_features,input_features)
        eval_gt_features, eval_input_features = unison_shuffled_copies(eval_gt_features, eval_input_features)
    
        # save features
        save_features('myData.h5py', input_features, eval_input_features, gt_features, eval_gt_features)

    
    # get the model and train
    # if not os.path.exists(model_name):
#     model = get_model(y_train.shape)

#     model_filename = 'SRCNN_{date:%Y-%m-%d %H:%M:%S}_best.h5'.format( date=datetime.datetime.now())
#     checkpoint = ModelCheckpoint(model_filename, monitor='val_loss', verbose=1, save_best_only=True,
#                                     save_weights_only=False, mode='min')
#     es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
#     callbacks_list = [checkpoint,es]
#     model.fit(X_train, y_train, batch_size=16, validation_data=(X_test, y_test),
#                     shuffle=True, epochs=200, callbacks=callbacks_list)

#     model.save('test-{date:%Y-%m-%d %H:%M:%S}.h5'.format( date=datetime.datetime.now() ))
    
    
    # predict and generate output files
    # model = load_model(model_filename)
    # y, fs = sf.read(input_filename)
    # reconstruct(y,fs,model)
    
    
if __name__ == "__main__":
    main()



#%%
