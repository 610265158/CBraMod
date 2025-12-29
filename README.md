<div align="center">

# EEG-Vision


_A solution from vision perspective to solve eeg problem_


Although numerous methods have been proposed to address EEG-related problems,
I believe that most existing works suffer from overclaiming issues and 
fail to capture the essential characteristics of EEG data. Based on this
observation, I developed this project that innovatively employs vision models
to tackle EEG problems.

This project builds upon the Cbramod framework and
conducts comprehensive comparative experiments across the 12 downstream tasks 
claimed in the Cbramod paper, obtaining corresponding experimental results.
The results will tell the story.


## 🔨 Setup
Install [Python](https://www.python.org/downloads/).

Install [PyTorch](https://pytorch.org/get-started/locally/).

Install other requirements:
```commandline
pip install -r requirements.txt
``` 


## 🚢 Train
```commandline
bash train_tuab.sh
```
We have released a pretrained checkpoint on [Hugginface🤗](https://huggingface.co/weighting666/CBraMod).



## 🚀 Quick Start
You can fine-tune the pretrained CBraMod on your custom downstream dataset using the following example code:


