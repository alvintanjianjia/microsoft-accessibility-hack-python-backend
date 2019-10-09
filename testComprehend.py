import json
import boto3

comprehend = boto3.client(service_name='comprehend')
text = "I don't want to live anymore"
print('Calling DetectSentiment')
response = json.dumps(comprehend.detect_sentiment(Text=text, LanguageCode='en'), sort_keys=True)
response = json.loads(response)

mixed = response['SentimentScore']['Mixed']
negative = response['SentimentScore']['Negative']
neutral = response['SentimentScore']['Neutral']
positive = response['SentimentScore']['Positive']

print('End of DetectSentiment\n')