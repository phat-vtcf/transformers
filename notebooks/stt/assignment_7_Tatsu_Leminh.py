#!/usr/bin/env python3

import torch
import librosa
import soundfile as sf

from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

def wav2vec_inference(wav_file):
    # load pretrained model
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

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
    print(wav2vec_inference('audio_samples/tatsu-wind-09092021.wav'))
