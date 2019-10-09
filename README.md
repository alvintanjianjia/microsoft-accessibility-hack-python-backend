# aws-hackdays-insideout-python-backend
This project runs the backend flask server, where it accepts an audio file, carries out analysis on the audio file, and returns the corresponding output results in a response.

## Prerequisites
1. Amazon AWS Account
2. PostgreSQL

## Setting Up
1. To rerun the model using another set of data, change the string pointing to the location of the raw data source.
2. Depending on the data source, certain preprocessing steps can be ommitted.


## Preprocessing Steps
1. Remove silence from the raw audio file.
2. Remove "Ellie's" voice from the audio file.
3. Segment the audio file into 10 second intervals using pyAudio.
4. Create spectograms for each interval.

## Analysis Steps
1. Audio is being passed to Amazon S3 (storage), Amazon Transcribe (speech-to-text), audio spectogram model.
2. The text that is returned is being passed to Amazon Comprehend (text based sentiment analysis) and local NLP module (psychological absolutist words).
3. The audio spectogram model would return a 1/0 classifier score (depressed / normal).
4. The scores returned from all 3 modules would be weighted and a final output score would be returned.
5. The results from each module would be saved into postgresql database. (Change URL / endpoint for cloud database).


## Built With
1. Tensorflow 1.13.0
2. scikit-learn
3. OpenCV

## Versioning
1.0

## Authors
Alvin Tan Jian Jia

## Acknowledgments
https://github.com/kykiefer/depression-detect
