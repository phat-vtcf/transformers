#!/usr/bin/env python3

import os
import torch
import librosa
import soundfile as sf

from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# load pretrained model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def wav2vec_inference(wav_file):

    # load audio
    audio_input, sample_rate = librosa.load(wav_file, sr=16000)

    # pad input values and return pt tensor
    input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values

    #INFERENCE

    # retrieve logits & take argmax
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)

    #transcribe
    transcription = processor.decode(predicted_ids[0])

    return transcription

if __name__ == '__main__':
    leminh_recording = 'audio_samples/leminh-wind-09092021.wav'
    tatsu_recording = 'audio_samples/tatsu-wind-09092021.wav'

    print()
    leminh_prediction = wav2vec_inference(leminh_recording)
    print(f'Prediction of {os.path.basename(leminh_recording)}: {leminh_prediction}')
    print()
    tatsu_prediction = wav2vec_inference(tatsu_recording)
    print(f'Prediction of {os.path.basename(tatsu_recording)}: {tatsu_prediction}')
