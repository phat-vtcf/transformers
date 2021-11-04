# VT Programming Assignment 7

This directory contains our inference reproduction to test the wav2vec Transformer model for the ASR task.

## Authors

- Tatsu Matsushima S4869214
- Leminh Nguyen S4923723

## Installation

To run the inference reproduction, you first need to install the required dependencies. It is recommended to install them in a virtual environment, this can be a [Conda](https://docs.conda.io/en/latest/) or a [Virtualenv](https://virtualenv.pypa.io/en/latest/) environment.

First install [Pytorch](https://pytorch.org/get-started/locally/) for your OS.

- M

Then install the remaining prerequisites:

```sh
python -m pip install -r requirements.txt
```

Run an instance of your favorite notebook enviroment and open the `assignment_7_Tatsu_Leminh.ipynb`.

Using Jupyter notebook:

```sh
jupyter notebook
```

Or upload it to [Google Colab](https://colab.research.google.com/) or [Deepnote](https://deepnote.com/dashboard).

##Original Repository
The repository we reproduced is the wav2vec Transformer model for the ASR task from Hagging Face. The model is based on Facebook's Wav2Vec2, which is pretrained and fine-tuned on 960 hours audio data with 16kHz from LibriSpeech.

The model acheived 1.8/3.3 WER on the clean test data sets in experiments by Hagging Face. wav2vec 2.0 with one hour labled data outperforms the previous state-of-art model on the 100 hour subset. Hence, the model demonstrates the high accuracy with limited amount of data.

##Our Task
Our task was to replicate the wav2vec Transformer model. We used the audio data of Leminh's dictation of the part of "The North Wind and the Sun" speech to experiment the model. For the result, we expect [].

##Outcomes
The output of the model achieved [] WER. This meets/did not meet our expectation in terms of [].
