import boto3

bucketName = "insideout-o1"
Key = "C:/Users/A\PycharmProjects/aws-hackdays-insideout/_73b96000000000001_77b36.wav"
outPutname = "C:/Users/A/PycharmProjects/aws-hackdays-insideout/_73b96000000000001_77b36.wav"

s3 = boto3.client('s3')
s3.upload_file(Key,bucketName,outPutname)