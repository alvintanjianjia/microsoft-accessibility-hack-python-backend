from __future__ import print_function
import tensorflow
import testPitch
import numpy
import os
import pandas
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
from keras.models import load_model
import cv2
from pydub import AudioSegment
import psycopg2
from sqlalchemy import create_engine
import time
import json
import boto3
import urllib
import skfuzzy
from nltk.parse import ShiftReduceParser
from nltk.parse import ShiftReduceParser, RecursiveDescentParser
import nltk
import nltk.data


def additional_nlp_score(transcribeText):
    # List of absolutist words
    absolutist_words = ['absolutely', 'all', 'always', 'complete', 'completely', 'constant', 'constantly',
                        'definitely', 'entire', 'ever', 'every', 'everyone', 'everything', 'full', 'must',
                        'never', 'nothing', 'totally', 'whole']
    transcribeTextlist = transcribeText.split(' ')
    absolutist_counter = 0
    for word in transcribeTextlist:
        if word in absolutist_words:
            absolutist_counter += 1
    return absolutist_counter/len(transcribeTextlist)


def sensibility_test(transcribeText, backdoor):
    if backdoor:
        print('Sentence is sensible')
        return 1
    else:
        grammar = nltk.data.load('grammars/book_grammars/drt.cfg')
        # sr = ShiftReduceParser(grammar=grammar)
        rd = RecursiveDescentParser()
        try:
            for t in rd.parse(transcribeText):
                print(t)
            print('Sentence is sensible')
            return 1
        except:
            print('Sentence is not sensible')
            return 0




def generateFinalResult(sensibility_test_score, mixed, neutral, positive, negative, personal_NLP_analytics_score,
                        prediction_value):
    score = (personal_NLP_analytics_score + prediction_value) /2
    if score >= 0.5:
        depression_value = 1
    else:
        depression_value = 0
    max_comprehend_score = max(mixed, neutral, positive, negative)
    confidence_value = (max_comprehend_score + sensibility_test_score + personal_NLP_analytics_score + prediction_value)/4

    return depression_value, confidence_value



# Convert Audio 3GP to WAV
def convertAudio3GP_wav(input_path, output_path):
    AudioSegment.converter = "C:\\Program Files\\ffmpeg\\ffmpeg\\bin\\ffmpeg.exe"
    AudioSegment.ffmpeg = "C:\\Program Files\\ffmpeg\\ffmpeg\\bin\\ffmpeg.exe"
    AudioSegment.ffprobe = "C:\\Program Files\\ffmpeg\\ffmpeg\\bin\\ffprobe.exe"

    ## converting .3gp file to .mp3 first
    inputFileName = input_path
    tempFileName = "C:/Users/A/PycharmProjects/aws-hackdays-insideout/upload_folder/testrecording.mp3"
    outputFileName = output_path
    sound = AudioSegment.from_file(inputFileName).export(tempFileName, format="mp3")
    sound = AudioSegment.from_mp3(tempFileName)
    sound.export(output_path, format="wav",
                 parameters=['-ar', '16000', '-ac', '1', '-ab', '64'])

# Get syllable count
def syllable_count(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count

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

# Plot Spectogram
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


def predict_from_spectogram(model_path='model_150_50.h5', img_path='D:/daizwoz_depression_dataset/data/img/test/depression/355_2.png'):
    model = load_model(model_path)

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    img = cv2.imread(img_path)
    img = cv2.resize(img, (150, 150))
    img = np.reshape(img, [1, 150, 150, 3])

    classes = model.predict_classes(img)
    if classes[0][0] == 1:
        print('normal')
        return 'normal'
    else:
        print('depressed')
        return 'depressed'


def sentimentAnalysis(text):
    comprehend = boto3.client(service_name='comprehend')

    print('Calling DetectSentiment')
    response = json.dumps(comprehend.detect_sentiment(Text=text, LanguageCode='en'), sort_keys=True)
    response = json.loads(response)

    mixed = response['SentimentScore']['Mixed']
    negative = response['SentimentScore']['Negative']
    neutral = response['SentimentScore']['Neutral']
    positive = response['SentimentScore']['Positive']

    print('End of DetectSentiment\n')

    return mixed, negative, neutral, positive

def insert_raw_results(user_id, sensibility_test_factor, amazon_comprehend_mixed_score,
                       amazon_comprehend_neutral_score, amazon_comprehend_positive_score,
                       amazon_comprehend_negative_score, personal_NLP_analytics_score,
                       audio_spectogram_model_score, final_result, transcribe_text, confidence_value):
    # sql_statement = """INSERT INTO results_raw(user_id, sensibility_test_factor, amazon_comprehend_mixed_score,
    #                    amazon_comprehend_neutral_score, amazon_comprehend_positive_score,
    #                    amazon_comprehend_negative_score, personal_NLP_analytics_score,
    #                    audio_spectogram_model_score, final_result) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    user = 'postgres'
    pwd = 'password123'
    host = 'insideout.c49fqzysd1id.us-east-1.rds.amazonaws.com'
    port = '5432'
    db = 'insideout'

    sql_statement = """INSERT INTO results_raw VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    try:
        connection = psycopg2.connect(user=user, password=pwd, host=host, port=port, database=db)
        cursor = connection.cursor()
        record_to_insert = (user_id, sensibility_test_factor, amazon_comprehend_mixed_score,
                       amazon_comprehend_neutral_score, amazon_comprehend_positive_score,
                       amazon_comprehend_negative_score, personal_NLP_analytics_score,
                       audio_spectogram_model_score, final_result, transcribe_text, confidence_value)
        cursor.execute(sql_statement, record_to_insert)
        connection.commit()
        print('Record inserted successfuly into mobile table.')
    except (Exception, psycopg2.Error) as error:
        print('Failed to insert record into raw_results table', error)


def audioToText(audioPath, jobName):
    bucketName = "insideout-o1"
    Key = audioPath
    outPutname = audioPath

    s3 = boto3.client('s3')
    s3.upload_file(Key, bucketName, outPutname)

    transcribe = boto3.client('transcribe')
    job_name = audioPath

    job_uri = "https://s3.us-east-1.amazonaws/insideout-o1/" + audioPath
    print(job_uri)


    transcribe.start_transcription_job(
            TranscriptionJobName=jobName,
            Media={'MediaFileUri': job_uri},
            MediaFormat='wav',
            LanguageCode='en-US')

    print('transcription job started')

    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=jobName)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break
        print("Not ready yet...")
        time.sleep(5)

    # print(status['TranscriptionJob']['Transcript']['TranscriptFileUri'])

    with urllib.request.urlopen(status['TranscriptionJob']['Transcript']['TranscriptFileUri']) as url:
        data = json.loads(url.read().decode())
        print(data)
        print(data['results']['transcripts'][0]['transcript'])

    transcribeText = data['results']['transcripts'][0]['transcript']

    return transcribeText






