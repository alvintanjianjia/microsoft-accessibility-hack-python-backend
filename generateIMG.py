import tensorflow
import testPitch
import numpy
import os
import pandas


import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks

""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)

""" scale frequency axis logarithmically """
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,int(scale[i]):], axis=1)
        else:
            newspec[:,i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]

    return newspec, freqs

""" plot spectrogram"""
def plotstft(audiopath, binsize=2**10, plotpath=None, colormap="jet"):
    samplerate, samples = wav.read(audiopath)

    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)

    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel

    timebins, freqbins = np.shape(ims)

    print("timebins: ", timebins)
    print("freqbins: ", freqbins)

    plt.figure(figsize=(3, 3))
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    plt.colorbar()

    # plt.xlabel("time (s)")
    # plt.ylabel("frequency (hz)")
    plt.xlim([0, timebins-1])
    plt.ylim([0, freqbins])

    xlocs = np.float32(np.linspace(0, timebins-1, 5))
    plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
    ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])

    if plotpath:
        plt.savefig(plotpath, bbox_inches="tight")
    # else:
        # plt.show()

    plt.cla()
    plt.close()

    return ims

def getSpectogram(filepath, dest='test.png'):
    plotstft(filepath, plotpath='test.png')

def generateSpectogram(filepath, dest='test.png'):
    plotstft(filepath, plotpath='test.png')

# getSpectogram('C:/Users/A/PycharmProjects/aws-hackdays-insideout/recording.wav')
# ims = plotstft('C:/Users/A/PycharmProjects/aws-hackdays-insideout/recording.wav')

if __name__ == '__main__':
    # directory containing participant folders with segmented wav files
    dir_name = 'D:/daizwoz_depression_dataset/data/segmented'
    df_train = pandas.read_csv('D:/daizwoz_depression_dataset/train.csv')
    df_test = pandas.read_csv('D:/daizwoz_depression_dataset/test.csv')
    os.makedirs('D:/daizwoz_depression_dataset/data/img/train/depression')
    os.makedirs('D:/daizwoz_depression_dataset/data/img/train/normal')
    os.makedirs('D:/daizwoz_depression_dataset/data/img/test/depression')
    os.makedirs('D:/daizwoz_depression_dataset/data/img/test/normal')
    train_depression_dir = 'D:/daizwoz_depression_dataset/data/img/train/depression'
    train_normal_dir = 'D:/daizwoz_depression_dataset/data/img/train/normal'
    test_depression_dir = 'D:/daizwoz_depression_dataset/data/img/test/depression'
    test_normal_dir = 'D:/daizwoz_depression_dataset/data/img/test/normal'
    # walks through wav files in dir_name and creates pngs of the spectrograms.
    # This is a visual representation of what is passed to the CNN before
    # normalization, although a cropped matrix representation is actually
    # passed.
    for subdir, dirs, files in os.walk(dir_name):
        for file in files:
            if file.endswith('.wav'):
                wav_file = os.path.join(subdir, file)
                # png_name = subdir + '/' + file[:-4] + '.png'
                segment = wav_file.split('_')[-1]
                segment = segment.split('.')[0]
                participantID = file.split('_')[0]
                participantID = participantID[1:]

                if int(participantID) in set(df_train['Participant_ID'].values):
                    if (df_train.loc[df_train['Participant_ID'] == int(participantID)]['PHQ8_Binary'] == 1).bool():
                        png_name = train_depression_dir + '/' + participantID + '_' + segment + '.png'
                        print('Processing ' + file + '...')
                        print(wav_file)
                        print(png_name)
                        try:
                            plotstft(wav_file, plotpath=png_name)
                        except:
                            os.remove(wav_file)
                            pass
                    elif (df_train.loc[df_train['Participant_ID'] == int(participantID)]['PHQ8_Binary'] == 0).bool():
                        png_name = train_normal_dir + '/' + participantID + '_' + segment + '.png'
                        print('Processing ' + file + '...')
                        print(wav_file)
                        print(png_name)
                        try:
                            plotstft(wav_file, plotpath=png_name)
                        except:
                            os.remove(wav_file)
                            pass

                elif int(participantID) in set(df_test['Participant_ID'].values):
                    if (df_test.loc[df_test['Participant_ID'] == int(participantID)]['PHQ8_Binary'] == 1).bool():
                        png_name = test_depression_dir + '/' + participantID + '_' + segment + '.png'
                        print('Processing ' + file + '...')
                        print(wav_file)
                        print(png_name)
                        try:
                            plotstft(wav_file, plotpath=png_name)
                        except:
                            os.remove(wav_file)
                            pass
                    elif (df_test.loc[df_test['Participant_ID'] == int(participantID)]['PHQ8_Binary'] == 0).bool():
                        png_name = test_normal_dir + '/' + participantID + '_' + segment + '.png'
                        print('Processing ' + file + '...')
                        print(wav_file)
                        print(png_name)
                        try:
                            plotstft(wav_file, plotpath=png_name)
                        except:
                            os.remove(wav_file)
                            pass









