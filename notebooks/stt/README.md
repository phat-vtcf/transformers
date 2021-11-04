# VT Programming Assignment 7

This directory contains our inference reproduction to test the wav2vec Transformer model for the ASR task.

## Authors

- Tatsu Matsushima S4869214
- Leminh Nguyen S4923723

## Installation

To run the inference reproduction, you have two possibilities:

1. You need to install the required dependencies. It is recommended to install them in a virtual environment, this can be a [Conda](https://docs.conda.io/en/latest/) or a [Virtualenv](https://virtualenv.pypa.io/en/latest/) environment.

First install [Pytorch](https://pytorch.org/get-started/locally/) for your OS.

Then install the remaining prerequisites:

```sh
python -m pip install -r requirements.txt
```

Run an instance of your favorite notebook enviroment and open the `assignment_7_Tatsu_Leminh.ipynb` notebook or execute the inference script:

```sh
python assignment_7_Tatsu_Leminh.py
```

2. Or open it directly in [Google Colab](https://colab.research.google.com/):

| Notebook                                                     | Description                                                  |                                                              |
| :----------------------------------------------------------- | :----------------------------------------------------------- | -----------------------------------------------------------: |
| [Inference of our Speechsounds North_wind recordings](https://colab.research.google.com/drive/1miIUmG_4x8fNbpJ0X8FrwxIJMm3SmUbx?usp=sharing) | Using the Wav2Vec2 Transformer model to convert our audio recording to text | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1miIUmG_4x8fNbpJ0X8FrwxIJMm3SmUbx?usp=sharing) |

## Original Repository

The original repository *'aims to make cutting-edge NLP easier to use for everyone'*. In this repository we reproduced the Speech-to-Text (STT) inference using the wav2vec2 Transformer model implemented in the Hugging Face library. The model is based on Facebook's Wav2Vec2, which is pretrained and fine-tuned on 960 hours audio data with 16kHz from LibriSpeech.

The model acheived 1.8/3.3 WER on the clean test data sets in experiments by Hugging Face. wav2vec 2.0 with one hour labled data outperforms the previous state-of-art model on the 100 hour subset. Hence, the model demonstrates the high accuracy with limited amount of data.

## Our Task

Our task was to reproduce an STT inference task using the wav2vec Transformer model. We used the audio sample of Leminh's and Tatsu's dictation of the "The North Wind and the Sun" text as the model input. You can see the full script for the speech dictation below.

### Script for the dictation

*"The North Wind and the Sun were disputing which was the stronger, when a traveller came along wrapped in a warm cloak. They agreed that the one who first succeeded in making the traveller take his cloak off should be considered stronger than the other. Then the North Wind blew as hard as he could, but the more he blew the more closely did the traveller fold his cloak around him; and at last the North Wind gave up the attempt. Then the Sun shined out warmly, and immediately the traveller took off his cloak. And so the North Wind was obliged to confess that the Sun was the stronger of the two."*

### Expectations for results
For the result, we expected that WERs for both audio inputs are lower than the original model since the model is trained with native speaker audio data. At the same time, we expected not siginifican high WER since we used the read speech data for testing. Also, Leminh learned English in the earlier age compared to Tatsu, thus we assumed that Leminh's WER is lower than Tatsu's WER. 

### Outcomes
The output of the model achieved 7.96 and 21.24 WER for Leminh's and Tatsu's speech. The calculation was done with JiWER module. The results meet our expectations being described above. You can see how the model performed with our audio inputs and how we calculated WER for each audio in the given notebook. 
