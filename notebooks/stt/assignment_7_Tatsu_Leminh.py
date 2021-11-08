#!/usr/bin/env python3

import os
import torch
import librosa

from jiwer import wer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# load pretrained model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def wav2vec_inference(wav_file):

    # load audio
    audio_input, sample_rate = librosa.load(wav_file, sr=16000)

    # tokenize audio input with the wav2vec2 transformer model
    input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values

    #INFERENCE

    # retrieve logits & take argmax
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)

    #transcribe prediction
    transcription = processor.decode(predicted_ids[0])

    return transcription

if __name__ == '__main__':

    # the original transcription of the read speech
    original_text = "THE NORTH WIND AND THE SUN WERE DISPUTING WHICH WAS THE STRONGER WHEN A TRAVELLER CAME ALONG WRAPPED IN A WARM CLOAK THEY AGREED THAT THE ONE WHO FIRST SUCCEEDED IN MAKING THE TRAVELLER TAKE HIS CLOAK OFF SHOULD BE CONSIDERED STRONGER THAN THE OTHER THEN THE NORTH WIND BLEW AS HARD AS HE COULD BUT THE MORE HE BLEW THE MORE CLOSELY DID THE TRAVELLER FOLD HIS CLOAK AROUND HIM AND AT LAST THE NORTH WIND GAVE UP THE ATTEMPT THEN THE SUN SHINED OUT WARMLY AND IMMEDIATELY THE TRAVELLER TOOK OFF HIS CLOAK AND SO THE NORTH WIND WAS OBLIGED TO CONFESS THAT THE SUN WAS THE STRONGER OF THE TWO"

    leminh_recording = 'audio_samples/leminh-wind-09092021.wav'
    tatsu_recording = 'audio_samples/tatsu-wind-09092021.wav'

    # get prediction of our audio recordings
    print()
    print(f'Original transcription:\n{original_text}')
    print()
    leminh_prediction = wav2vec_inference(leminh_recording)
    print(f'Prediction of {os.path.basename(leminh_recording)}:\n{leminh_prediction}')
    print()
    tatsu_prediction = wav2vec_inference(tatsu_recording)
    print(f'Prediction of {os.path.basename(tatsu_recording)}:\n{tatsu_prediction}')
    print()

    # WER is calculated in percentage.
    WER_leminh = round(wer(original_text, leminh_prediction)*100, 2)
    WER_tatsu = round(wer(original_text, tatsu_prediction)*100, 2)

    print(f'WER_leminh: {WER_leminh}% WER')
    print(f'WER_tatsu: {WER_tatsu}% WER')
