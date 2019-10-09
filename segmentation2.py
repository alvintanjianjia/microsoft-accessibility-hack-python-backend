import os
from pydub import AudioSegment
from pydub.utils import make_chunks
import math

def round_down(num, divisor):
    return num - (num%divisor)

# directory containing raw wav files
dir_name = 'D:/daizwoz_depression_dataset/data/interim'

# directory where a participant folder will be created containing their
# segmented wav file
out_dir = 'D:/daizwoz_depression_dataset/data/interim'
new_dir = 'D:/daizwoz_depression_dataset/data/segmented'

# iterate through wav files in dir_name and create a segmented wav_file
for file in os.listdir(dir_name):
    current_dir = dir_name + '/' + str(file)
    for file2 in os.listdir(current_dir):
        current_file = current_dir + '/' + str(file2)
        new_file = new_dir + '/' + str(file) + '/' + str(file2)
        # print(new_file, 'newfile')
        # print(current_file)
        counter = 0
        newAudio = AudioSegment.from_wav(current_file)
        total_sec = round_down(newAudio.duration_seconds, 10)
        print(total_sec)
        last = total_sec / 10

        chunk_length_ms = 10000
        chunks = make_chunks(newAudio, chunk_length_ms)

        for i, chunk in enumerate(chunks):
            chunk_name = new_dir + '/' + str(file2).split('_')[0] + '_' + str(i) + '.wav'
            print(chunk_name)
            chunk.export(chunk_name, format='wav')

        # while counter < total_sec:
        #     t1 = (counter) * 1000
        #     t2 = (counter + 10) * 1000
        #     print(t1, t2, 't1 & t2')
        #     newAudio = newAudio[t1:t2]
        #     new_filename = str(file2).split('_')[0] + '_' + str(t1) + '_' + str(t2)
        #     counter += 10
        #     print(new_dir + '/' + new_filename)
        #     newAudio.export(new_dir + '/' + new_filename + '.wav', format='wav')




