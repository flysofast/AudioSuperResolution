import subprocess
import os

for filename in os.listdir(os.getcwd()):
    if (filename.endswith(".wav")):
        name = filename.split(".")[0]
        os.system("ffmpeg -i {0}.wav -ab 96000 {1}.mp3".format(name, name))
        os.system("ffmpeg -i {0}.mp3 {1}.wav".format(name, name+"down"))
        os.remove("{0}.mp3".format(name))



