
from pydub import AudioSegment

def convertAudio3GP_wav(input_path, output_path):
    AudioSegment.converter = "C:\\Program Files\\ffmpeg\\ffmpeg\\bin\\ffmpeg.exe"
    AudioSegment.ffmpeg = "C:\\Program Files\\ffmpeg\\ffmpeg\\bin\\ffmpeg.exe"
    AudioSegment.ffprobe = "C:\\Program Files\\ffmpeg\\ffmpeg\\bin\\ffprobe.exe"

    ## converting .3gp file to .mp3 first
    inputFileName = input_path
    tempFileName = "C:/Users/A/PycharmProjects/aws-hackdays-insideout/upload_folder/testrecording.mp3"
    outputFileName = output_path
    sound = AudioSegment.from_file(inputFileName).export(outputFileName, format="mp3")
    sound = AudioSegment.from_mp3(tempFileName)
    sound.export(output_path, format="wav",
                 parameters=['-ar', '16000', '-ac', '1', '-ab', '64'])


AudioSegment.converter = "C:\\Program Files\\ffmpeg\\ffmpeg\\bin\\ffmpeg.exe"
AudioSegment.ffmpeg = "C:\\Program Files\\ffmpeg\\ffmpeg\\bin\\ffmpeg.exe"
AudioSegment.ffprobe ="C:\\Program Files\\ffmpeg\\ffmpeg\\bin\\ffprobe.exe"

## converting .3gp file to .mp3 first
inputFileName = 'C:/Users/A/PycharmProjects/aws-hackdays-insideout/upload_folder/testfile.3gp'
outputFileName = 'C:/Users/A/PycharmProjects/aws-hackdays-insideout/testrecording.mp3'
sound = AudioSegment.from_file(inputFileName).export(outputFileName, format="mp3")
sound = AudioSegment.from_mp3("C:/Users/A/PycharmProjects/aws-hackdays-insideout/testrecording.mp3")
sound.export('C:/Users/A/PycharmProjects/aws-hackdays-insideout/recording.wav', format="wav", parameters=['-ar', '16000', '-ac', '1', '-ab', '64'])