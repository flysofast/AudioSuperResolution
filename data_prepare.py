#%%
import os
from pydub import AudioSegment
import subprocess

#%%
"""
This module is used to prepare the training data (in data_input folder)
"""

#%%
def stereo_to_mono(input_dir, output_dir):
    """
    Check for stereo audio files in input folder, then 
    convert them to mono and put them in the output folder and 
    """
    if os.path.exists(output_dir):
        print("{} already exist".format(output_dir))
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            # print(os.path.join(input_dir, filename))
            sound = AudioSegment.from_wav(os.path.join(input_dir, filename))
            if sound.channels != 1:
                sound = sound.set_channels(1)
            name = filename.split(".")[0]
            sound.export(os.path.join(output_dir, name) + ".wav", format="wav")

    print("-----------Done converting Stereo to Mono---------------")

#%%
def compress(input_dir, output_dir):
    """
    Compress the wav files in the data_mono folder to mp3, then convert 
    them back to wav and put them in data_input folder.
    """

    if os.path.exists(output_dir):
        print("{} already exist".format(output_dir))
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # change cwd to input_dir and compress the files
    os.chdir(input_dir) 
    for filename in os.listdir(os.path.join(os.getcwd(), input_dir)):
        if (filename.endswith(".wav")):
            name = filename.split(".")[0]
            os.system("ffmpeg -i {0}.wav -ab 96000 {1}.mp3".format(name, name))
            # move the files to output_dir
            os.rename(os.path.join(input_dir, "{}.mp3".format(name)), \
                        os.path.join(output_dir, "{}.mp3".format(name)))
    
    # change cwd to output_dir to convert mp3s back to wavs the files
    os.chdir(output_dir) 
    for filename in os.listdir(os.path.join(os.getcwd(), input_dir)):
        name = filename.split(".")[0]
        os.system("ffmpeg -i {0}.mp3 {1}.wav".format(name, name))
        os.remove("{0}.mp3".format(name))

    print("-----------Done Compressing---------------")