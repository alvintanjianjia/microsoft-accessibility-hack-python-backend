from flask import Flask, request, render_template, jsonify
from json import dumps
import requests
import json
import urllib.request as urllib
import os
from function import *
import tensorflow as tf
import cv2
import time, datetime

global graph
graph = tf.get_default_graph()
model_path='model_150_50.h5'
model = load_model(model_path)

# Turn on and off transcribe
global transcribe
transcribe = True

# Change for backdoor
global backdoor, backdoor_counter, backdoor_string1, backdoor_string2
backdoor = True
backdoor_counter = 0
backdoor_string1 = 'Hi, how are you feeling today?'
backdoor_string2 = 'Is there anything that you want to talk to me about?'
# backdoor_string2 = 'Did something good happen today?'
# model.compile(loss='binary_crossentropy',
#                 optimizer='rmsprop',
#                 metrics=['accuracy'])
UPLOAD_FOLDER = 'C:/Users/A/PycharmProjects/aws-hackdays-insideout/upload_folder'

ALLOWED_EXTENSIONS = set(['3gp'])

app = Flask(__name__)



app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/", methods = ['POST', 'GET'])
def testServerRest():
    paranoidStatement = "server up and running - lol Alvin is paranoid"
    print(paranoidStatement)
    return(paranoidStatement)


@app.route("/sendAudio/", methods = ['POST', 'GET'])
def getAudio():
    print("starting send audio")
    current_timestamp = time.mktime(datetime.datetime.today().timetuple())
    awsFileName = str(current_timestamp).split('.')[0]
    if request.method == 'POST':
        int_message = 1
        print("Data uploading")
        print(request.headers)
        for v in request.values:
            print(v)
        # logdata = request.stream.readline()
        # while(logdata):
        #    print "uploading"
        #    print logdata
        #    logdata = request.stream.readline()

        # print(request.files)
        file = request.files['uploadedfile']
        # print(file)
        outputFilePath = 'C:/Users/A/PycharmProjects/aws-hackdays-insideout/upload_folder/' + str(current_timestamp).split('.')[0] + '.wav'
        pngPath = 'C:/Users/A/PycharmProjects/aws-hackdays-insideout/upload_folder/' + str(current_timestamp).split('.')[0] + '.png'
        file.save('C:/Users/A/PycharmProjects/aws-hackdays-insideout/upload_folder/' + str(current_timestamp).split('.')[0] + '.3gp')
        audioPath = outputFilePath
        convertAudio3GP_wav(input_path='C:/Users/A/PycharmProjects/aws-hackdays-insideout/upload_folder/' + str(current_timestamp).split('.')[0] + '.3gp',
                            output_path=outputFilePath)
        print("Uploading done")

        plotstft(outputFilePath, plotpath=pngPath)
        # with graph.as_default():
        with graph.as_default():
            img = cv2.imread(pngPath)
            img = cv2.resize(img, (150, 150))
            img = np.reshape(img, [1, 150, 150, 3])
            classes = model.predict_classes(img)
        if classes[0][0] == 1:
            # Normal = prediction_value==0, classes[0][0]==1
            prediction_value = 0
            print('Prediction based on spectogram = NORMAL')
        else:
            # Depressed = prediction_value==1, classes[0][0]==0
            prediction_value = 1
            print('Prediction based on spectogram = DEPRESSED')

        try:
            #Amazon Transcribe
            # transcribeText = audioToText(audioPath, awsFileName)
            transcribeText = 'Ommiting transcribe text'
            print(transcribeText)

            #Amazon Comprehend
            mixed, negative, neutral, positive = sentimentAnalysis(transcribeText)
            print('Mixed Value: ' + str(mixed))
            print('Negative Value: ' + str(negative))
            print('Neutral Value: ' + str(neutral))
            print('Positive Value: ' + str(positive))

            #Sensibility test / module
            sensibility_test_score = sensibility_test(transcribeText,backdoor=True)


            #Personal NLP + analytics
            personal_NLP_analytics_score = additional_nlp_score(transcribeText)






        except:
            print('transcribe failed')
    # return Response(str(int_message), mimetype='text/plain')

    #Generate Final Result
    depression_value, confidence_value = generateFinalResult(sensibility_test_score, mixed, neutral, positive, negative,
                                                             personal_NLP_analytics_score,prediction_value)


    insert_raw_results('test', sensibility_test_score, mixed, neutral, positive, negative,
                       personal_NLP_analytics_score,
                       prediction_value, depression_value, transcribeText, confidence_value)

    if backdoor:
        global backdoor_counter
        if backdoor_counter == 0:
            backdoor_counter += 1
            return backdoor_string1
        elif backdoor_counter == 1:
            return backdoor_string2
    else:

        return (prediction_value)



app.run(host = '192.168.0.135')