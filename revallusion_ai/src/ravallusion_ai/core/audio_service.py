import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()

import os
from moviepy.video.io.VideoFileClip import VideoFileClip
from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)

import numpy as np
from numpy.linalg import norm

from sentence_transformers import SentenceTransformer
EMBEDDING_MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")


def extract_audio_from_video(video_path:str, save_path:str) -> None:
    
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(save_path)
    return save_path

def transcribe_audio(audio_path:str) -> str:

    deepgram = DeepgramClient(os.getenv('DEEPGRAM_API_KEY'))
    with open(audio_path, "rb") as file:
        buffer_data = file.read()
    payload: FileSource = {
        "buffer": buffer_data,
    }
    options = PrerecordedOptions(model="nova-2",)
    response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)
    transcription = eval(response.to_json(indent=4))
    transcript = transcription['results']['channels'][0]['alternatives'][0]['transcript']
    return transcript


# metrics

def cosine_similarity(vec_1,vec_2):
    np_vec_1 = np.array(vec_1)
    np_vec_2 = np.array(vec_2)
    similarity = np.dot(np_vec_1,np_vec_2)/(norm(np_vec_1)*norm(np_vec_2))
    return similarity * 100
