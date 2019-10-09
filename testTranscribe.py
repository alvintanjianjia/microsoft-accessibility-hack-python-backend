from __future__ import print_function
import time
import json
import boto3
import urllib

transcribe = boto3.client('transcribe')
job_name = "15506847041"
job_uri = "https://s3.us-east-1.amazonaws/insideout-o1/C:/Users/A/PycharmProjects/aws-hackdays-insideout/upload_folder/1550684704.wav"




def start_transcription_job():
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': job_uri},
        MediaFormat='wav',
        LanguageCode='en-US'
    )

start_transcription_job()

def get_transcription_job(job_name):
    transcribe.get_transcription_job(
        TranscriptionJobName=job_name
    )


while True:
    status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
    if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
        break
    print("Not ready yet...")
    time.sleep(5)

# print(status['TranscriptionJob']['Transcript']['TranscriptFileUri'])

with urllib.request.urlopen(status['TranscriptionJob']['Transcript']['TranscriptFileUri']) as url:
    data = json.loads(url.read().decode())
    print(data)
    print(data['results']['transcripts'][0]['transcript'])