import torch
from pyannote.audio import Pipeline
import time
from pydub import AudioSegment
import python_speech_to_text_v2
from io import BytesIO
import numpy as np
import datetime
import python_segmenter
from google.cloud import storage
import python_vertexai
import python_text_to_speech
from google.cloud import speech_v1p1beta1 as speech
import python_speech_to_text
from moviepy.editor import AudioFileClip, VideoFileClip

gcs_client = storage.Client.from_service_account_json(".\\ci2-graphical-part-dev-353b03aed605.json")
input_file = ".\\separated\\htdemucs\\blog_GanAI\\vocals.wav"
dt_now = datetime.datetime.now()


def speech_to_text_and_transcript(param):
    newAudio = AudioSegment.from_wav(input_file)
    newAudio = newAudio[param["start_time"]: param["end_time"]]
    newAudio = newAudio.set_channels(1)
    print(f"オーディオ時間：{newAudio.duration_seconds}")
    output = BytesIO()
    newAudio.export(output , format="wav")
    file_name = f"audio/vocals_{dt_now.strftime('%Y%m%d%H%M%S')}.wav"
    bucket = gcs_client.bucket('speech-to-text-example-20230625')
    blob = bucket.blob(file_name)
    blob.upload_from_string(output.getvalue())
    audio = speech.RecognitionAudio(uri="gs://speech-to-text-example-20230625/" + file_name)
    speech_to_text = python_speech_to_text.get_speech_to_text(audio)
    '''
    if newAudio.duration_seconds > 55:
        file_name = f"vocals_{dt_now.strftime('%Y%m%d%H%M%S')}.wav"
        bucket = gcs_client.bucket('speech-to-text-example-20230625')
        blob = bucket.blob(file_name)
        blob.upload_from_string(output.getvalue())
        speech_to_text = python_speech_to_text_v2.get_speech_to_text_v2("ci2-graphical-part-dev", "test1", "gs://speech-to-text-example-20230625/" + file_name)
    else:
        speech_to_text = python_speech_to_text_v2.get_speech_to_text_v2_by_short("ci2-graphical-part-dev", "test1", output.getvalue())
    '''
    
    recode = python_speech_to_text.get_translation(speech_to_text)
    #newAudio.export(param["output_file"] , format="wav")
    return recode

def get_pipeline(diarization):
    counter = 0
    result_list = []
    speaker_data = ""
    start_time = 0
    end_time = 0
    output_file = ""
    list_data = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        output_file = ".\\output\\pyannote4\\"+ f"speaker_{counter}_{speaker}" + ".wav"
        start_time = turn.start * 1000
        end_time = turn.end * 1000
        diff_time = end_time - start_time
        #print(f"{start_time / 1000} {end_time / 1000} {diff_time / 1000} {output_file}")
        list_data.append({
            "speaker": speaker,
            "start_time": start_time,
            "end_time": end_time
        })
    print(list_data)

    '''
    list_data = [{'speaker': 'SPEAKER_00', 'start_time': 930.0341296928328, 'end_time': 4718.430034129692}, {'speaker': 'SPEAKER_00', 'start_time': 6220.136518771331, 'end_time': 20145.05119453925}, {'speaker': 'SPEAKER_00', 'start_time': 21237.201365187717, 'end_time': 27500.0}, {'speaker': 'SPEAKER_00', 'start_time': 28199.658703071673, 'end_time': 64667.23549488056}, {'speaker': 'SPEAKER_01', 'start_time': 65588.73720136519, 'end_time': 133131.39931740615}, {'speaker': 'SPEAKER_01', 'start_time': 133933.44709897612, 'end_time': 139940.27303754268}, {'speaker': 'SPEAKER_00', 'start_time': 142875.42662116044, 'end_time': 151476.10921501706}, {'speaker': 'SPEAKER_00', 'start_time': 152738.90784982938, 'end_time': 166680.88737201368}, {'speaker': 'SPEAKER_02', 'start_time': 167892.4914675768, 'end_time': 167994.8805460751}, {'speaker': 'SPEAKER_02', 'start_time': 179496.58703071674, 'end_time': 180759.385665529}, {'speaker': 'SPEAKER_02', 'start_time': 181544.36860068262, 'end_time': 183506.8259385666}, {'speaker': 'SPEAKER_02', 'start_time': 189991.46757679182, 'end_time': 190008.53242320818}, {'speaker': 'SPEAKER_02', 'start_time': 195503.41296928326, 'end_time': 207500.0}, {'speaker': 'SPEAKER_01', 'start_time': 202994.8805460751, 'end_time': 203711.60409556312}, {'speaker': 'SPEAKER_02', 'start_time': 214001.70648464165, 'end_time': 220503.41296928326}, {'speaker': 'SPEAKER_02', 'start_time': 228506.8259385666, 'end_time': 242226.96245733788}, {'speaker': 'SPEAKER_02', 'start_time': 243063.13993174065, 'end_time': 243114.33447098976}, {'speaker': 'SPEAKER_02', 'start_time': 247704.77815699662, 'end_time': 249325.9385665529}, {'speaker': 'SPEAKER_00', 'start_time': 249906.1433447099, 'end_time': 254052.9010238908}, {'speaker': 'SPEAKER_02', 'start_time': 254957.33788395903, 'end_time': 267994.88054607506}, {'speaker': 'SPEAKER_02', 'start_time': 271066.5529010239, 'end_time': 271544.3686006826}, {'speaker': 'SPEAKER_02', 'start_time': 272244.02730375424, 'end_time': 274496.58703071676}, {'speaker': 'SPEAKER_02', 'start_time': 284496.58703071676, 'end_time': 287329.3515358362}, {'speaker': 'SPEAKER_00', 'start_time': 291032.4232081911, 'end_time': 300947.0989761092}, {'speaker': 'SPEAKER_00', 'start_time': 301646.7576791809, 'end_time': 314940.2730375427}, {'speaker': 'SPEAKER_03', 'start_time': 316510.2389078498, 'end_time': 427755.97269624576}, {'speaker': 'SPEAKER_00', 'start_time': 428080.204778157, 'end_time': 433916.38225255976}, {'speaker': 'SPEAKER_03', 'start_time': 433916.38225255976, 'end_time': 474035.8361774744}, {'speaker': 'SPEAKER_00', 'start_time': 478046.07508532424, 'end_time': 499906.1433447099}, {'speaker': 'SPEAKER_04', 'start_time': 500622.86689419794, 'end_time': 583216.723549488}, {'speaker': 'SPEAKER_00', 'start_time': 583575.0853242321, 'end_time': 588199.6587030716}, {'speaker': 'SPEAKER_04', 'start_time': 588967.5767918089, 'end_time': 601817.4061433447}, {'speaker': 'SPEAKER_04', 'start_time': 602448.8054607509, 'end_time': 640247.4402730375}, {'speaker': 'SPEAKER_00', 'start_time': 640691.1262798635, 'end_time': 648831.0580204779}]
    '''
    
    #list = connect_time(list_data)
    #print(list)
    
    for i, data in enumerate(list_data):
        diff_time = data["end_time"] - data["start_time"]
        #print(f"{data[1] / 1000} {data[2] / 1000} {diff_time / 1000} {data[0]}")
    
        if speaker_data == "":
            speaker_data = data["speaker"]
            start_time = data["start_time"]
            end_time = data["end_time"]
            continue

        if speaker_data == data["speaker"] and not i == len(list_data) - 1:
            end_time = data["end_time"]
            continue
        else:
            output_file = ".\\output\\pyannote2\\"+ f"speaker_{counter}_{speaker_data}" + ".wav"
            newAudio = AudioSegment.from_wav(input_file)
            newAudio = newAudio[start_time: end_time]
            newAudio.export(output_file , format="wav")
            segment_list = python_segmenter.get_segmenter_list(output_file)
            print(segment_list)
            counter2 = 0
            for segment in segment_list:
                output_file = ".\\output\\pyannote3\\"+ f"speaker_{counter}_{counter2}_{speaker_data}" + ".wav"
                counter2 = counter2 + 1
                #segment[3].export(output_file , format="wav")
                result = {
                    "speaker": speaker_data,
                    "start_time": start_time + segment[1] * 1000,
                    "end_time": start_time + segment[2] * 1000,
                    "output_file": output_file,
                    "message_transcript": "",
                    "message_translation": ""
                }
                recode = speech_to_text_and_transcript(result)
                result["message_transcript"] = recode["transcript"]
                result["message_translation"] = recode["translation"]
                if recode["translation"] == "":
                    continue
                print(result)
                result_list.append(result)
            counter = counter + 1
            speaker_data = data["speaker"]
            start_time = data["start_time"]
            end_time = data["end_time"]
            #print(result_list)
    
    output_file = ".\\output\\pyannote2\\"+ f"speaker_{counter}_{speaker_data}" + ".wav"
    newAudio = AudioSegment.from_wav(input_file)
    newAudio = newAudio[start_time: end_time]
    newAudio.export(output_file , format="wav")
    segment_list = python_segmenter.get_segmenter_list(output_file)
    print(segment_list)
    counter2 = 0
    for segment in segment_list:
        output_file = ".\\output\\pyannote3\\"+ f"speaker_{counter}_{counter2}_{speaker_data}" + ".wav"
        counter2 = counter2 + 1
        #segment[3].export(output_file , format="wav")
        result = {
            "speaker": speaker_data,
            "start_time": start_time + segment[1] * 1000,
            "end_time": start_time + segment[2] * 1000,
            "output_file": output_file,
            "message_transcript": "",
            "message_translation": ""
        }
        recode = speech_to_text_and_transcript(result)
        result["message_transcript"] = recode["transcript"]
        result["message_translation"] = recode["translation"]
        if recode["translation"] == "":
            continue
        print(result)
        result_list.append(result)
    print(result_list)
    np.savetxt('pyannote_'  + dt_now.strftime('%Y%m%d%H%M%S') +  '.csv', result_list, fmt = "%s", delimiter=',')
    return result_list

'''
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@develop", use_auth_token="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")
pipeline=pipeline.to(torch.device('cuda'))

time_sta = time.time()
diarization = pipeline(input_file)
time_end = time.time()

tim = time_end - time_sta
print(tim)
result_list = get_pipeline(diarization)
'''

speech_to_text_data = [{'speaker': 'SPEAKER_00', 'start_time': 5008.532423208191, 'end_time': 8008.532423208191, 'output_file': '.\\output\\pyannote3\\speaker_0_0_SPEAKER_00.wav', 'message_transcript': 'それでは処理フローについてご説明します', 'message_translation': 'Now, I will explain the processing flow.'}, {'speaker': 'SPEAKER_00', 'start_time': 9028.532423208191, 'end_time': 11968.532423208191, 'output_file': '.\\output\\pyannote3\\speaker_0_1_SPEAKER_00.wav', 'message_transcript': 'まずは音声と映像に分離します', 'message_translation': 'First, separate audio and video'}, {'speaker': 'SPEAKER_00', 'start_time': 13148.532423208191, 'end_time': 17768.53242320819, 'output_file': '.\\output\\pyannote3\\speaker_0_2_SPEAKER_00.wav', 'message_transcript': '分離された音声よ人の声と BGM にそれぞれ分離します', 'message_translation': 'Separated audio is separated into human voice and BGM.'}, {'speaker': 'SPEAKER_00', 'start_time': 19188.53242320819, 'end_time': 30008.53242320819, 'output_file': '.\\output\\pyannote3\\speaker_0_3_SPEAKER_00.wav', 'message_transcript': '釣りには処分良子なって行きますうちらは初場所を好みさせるのと発音が終わるタイミングを識別する目的で利用しています', 'message_translation': 'When it comes to fishing, I use it for the purpose of getting people to prefer the first place and identifying when the pronunciation ends.'}, {'speaker': 'SPEAKER_00', 'start_time': 31688.53242320819, 'end_time': 35188.53242320819, 'output_file': '.\\output\\pyannote3\\speaker_0_4_SPEAKER_00.wav', 'message_transcript': 'そ の後音声認識を抱く行います', 'message_translation': 'Then do the voice recognition hug'}, {'speaker': 'SPEAKER_00', 'start_time': 36548.53242320819, 'end_time': 42168.5324232082, 'output_file': '.\\output\\pyannote3\\speaker_0_5_SPEAKER_00.wav', 'message_transcript': '翻訳された文章をもとに音声合成としゃく合わせを行います', 'message_translation': 'Performs speech synthesis and matching based on the translated text'}, {'speaker': 'SPEAKER_00', 'start_time': 43387.37201365188, 'end_time': 48567.37201365188, 'output_file': '.\\output\\pyannote3\\speaker_1_0_SPEAKER_00.wav', 'message_transcript': '最後に映像と音声を合成して動画を作成します', 'message_translation': 'Finally, combine the video and audio to create a video.'}]
#speech_to_text_data = result_list
list = []
counter = 0
for item in speech_to_text_data:
    list.append({
        "gender": "male",
        "start_time": float(item["start_time"]) / 1000,
        "end_time": float(item["end_time"]) / 1000,
        "message_en": item["message_translation"],
        "message_ja": item["message_transcript"],
    })

audio = python_text_to_speech.text_to_speech_exe(list, float(speech_to_text_data[0]["start_time"]) / 1000)
translation_path = f".\\output\\translation\\translation_audio_{dt_now.strftime('%Y%m%d%H%M%S')}.wav"
audio.export( translation_path, format="wav")

input_vocals_file = ".\\separated\\htdemucs\\blog_GanAI\\vocals.wav"
input_no_vocals_file = ".\\separated\\htdemucs\\blog_GanAI\\no_vocals.wav"
export_vocal = AudioSegment.from_file(translation_path)
no_vocals = AudioSegment.from_file(input_no_vocals_file)
merge_audio = AudioSegment.from_file(input_vocals_file)
merge_audio = merge_audio.overlay(export_vocal, position=float(speech_to_text_data[0]["start_time"]), gain_during_overlay=-100)

wav_path = ".\\separated\\htdemucs\\blog_GanAI\\mixed_sounds.wav"
output = no_vocals.overlay(merge_audio, position=0)
output.export(wav_path, format="wav")

# MP4ファイルとWAVファイルのパスを指定
mp4_path = ".\\blog_GanAI.mp4"

# MP4ファイルとWAVファイルを読み込み
video_clip = VideoFileClip(mp4_path)
audio_clip = AudioFileClip(wav_path)


# WAVファイルの音声をMP4ファイルに合成
video_clip = video_clip.set_audio(audio_clip)

# 出力ファイル名とフォーマットを指定
output_filename = "new_blog_GanAI.mp4"
output_format = "mp4"

# 合成した動画を保存
video_clip.write_videofile(output_filename, audio_codec="aac", fps=video_clip.fps)



'''
result_list = []
list_data = [{'speaker': 'SPEAKER_00', 'start_time': 930.0341296928328, 'end_time': 64667.23549488056, 'output_file': '.\\output\\pyannote\\speaker_0_SPEAKER_00.wav', 'message_transcript': '', 'message_translation': ''},{'speaker': 'SPEAKER_01', 'start_time': 65588.73720136519, 'end_time': 139940.27303754268, 'output_file': '.\\output\\pyannote\\speaker_1_SPEAKER_01.wav', 'message_transcript': '', 'message_translation': ''},{'speaker': 'SPEAKER_00', 'start_time': 142875.42662116044, 'end_time': 166680.88737201368, 'output_file': '.\\output\\pyannote\\speaker_2_SPEAKER_00.wav', 'message_transcript': '', 'message_translation': ''},{'speaker': 'SPEAKER_02', 'start_time': 167892.4914675768, 'end_time': 207500.0, 'output_file': '.\\output\\pyannote\\speaker_3_SPEAKER_02.wav', 'message_transcript': '', 'message_translation': ''},{'speaker': 'SPEAKER_01', 'start_time': 202994.8805460751, 'end_time': 203711.60409556312, 'output_file': '.\\output\\pyannote\\speaker_4_SPEAKER_01.wav', 'message_transcript': '', 'message_translation': ''},{'speaker': 'SPEAKER_02', 'start_time': 214001.70648464165, 'end_time': 249325.9385665529, 'output_file': '.\\output\\pyannote\\speaker_5_SPEAKER_02.wav', 'message_transcript': '', 'message_translation': ''},{'speaker': 'SPEAKER_00', 'start_time': 249906.1433447099, 'end_time': 254052.9010238908, 'output_file': '.\\output\\pyannote\\speaker_6_SPEAKER_00.wav', 'message_transcript': '', 'message_translation': ''},{'speaker': 'SPEAKER_02', 'start_time': 254957.33788395903, 'end_time': 287329.3515358362, 'output_file': '.\\output\\pyannote\\speaker_7_SPEAKER_02.wav', 'message_transcript': '', 'message_translation': ''},{'speaker': 'SPEAKER_00', 'start_time': 291032.4232081911, 'end_time': 314940.2730375427, 'output_file': '.\\output\\pyannote\\speaker_8_SPEAKER_00.wav', 'message_transcript': '', 'message_translation': ''},{'speaker': 'SPEAKER_03', 'start_time': 316510.2389078498, 'end_time': 427755.97269624576, 'output_file': '.\\output\\pyannote\\speaker_9_SPEAKER_03.wav', 'message_transcript': '', 'message_translation': ''},{'speaker': 'SPEAKER_00', 'start_time': 428080.204778157, 'end_time': 433916.38225255976, 'output_file': '.\\output\\pyannote\\speaker_10_SPEAKER_00.wav', 'message_transcript': '', 'message_translation': ''},{'speaker': 'SPEAKER_03', 'start_time': 433916.38225255976, 'end_time': 474035.8361774744, 'output_file': '.\\output\\pyannote\\speaker_11_SPEAKER_03.wav', 'message_transcript': '', 'message_translation': ''},{'speaker': 'SPEAKER_00', 'start_time': 478046.07508532424, 'end_time': 499906.1433447099, 'output_file': '.\\output\\pyannote\\speaker_12_SPEAKER_00.wav', 'message_transcript': '', 'message_translation': ''},{'speaker': 'SPEAKER_04', 'start_time': 500622.86689419794, 'end_time': 583216.723549488, 'output_file': '.\\output\\pyannote\\speaker_13_SPEAKER_04.wav', 'message_transcript': '', 'message_translation': ''},{'speaker': 'SPEAKER_00', 'start_time': 583575.0853242321, 'end_time': 588199.6587030716, 'output_file': '.\\output\\pyannote\\speaker_14_SPEAKER_00.wav', 'message_transcript': '', 'message_translation': ''},{'speaker': 'SPEAKER_04', 'start_time': 588967.5767918089, 'end_time': 640247.4402730375, 'output_file': '.\\output\\pyannote\\speaker_15_SPEAKER_04.wav', 'message_transcript': '', 'message_translation': ''},{'speaker': 'SPEAKER_00', 'start_time': 640691.1262798635, 'end_time': 648831.0580204779, 'output_file': '.\\output\\pyannote\\speaker_16_SPEAKER_00.wav', 'message_transcript': '', 'message_translation': ''}]
for list in list_data:
    newAudio = AudioSegment.from_wav(list["output_file"])
    recode = speech_to_text(newAudio)
    list["message_transcript"] = recode["transcript"]
    list["message_translation"] = recode["translation"]
    result_list.append(list)

np.savetxt('pyannote_'  + dt_now.strftime('%Y%m%d%H%M%S') +  '.csv', result_list, fmt = "%s", delimiter=',')
'''
