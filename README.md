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



## 🚀 Result
Here it goes,
### 1.1 chb-mit

| model   | acc     | auc     | pr      | roc     |
|---------|---------|---------|---------|---------|
| effnet-b0 | 0.70594 | 0.42923 | 0.89768 | -       |

### tuab
 model   | acc     | auc            | pr      | roc     |
|---------|---------|----------------|---------|---------|
| effnet-b0 | 0.82049| 0.89364| 0.89177 |    
| convnextv2_small | 0.83136| 0.91319| 0.90551 |     

### tuev
model   | acc     | auc            | pr      | roc     |
|---------|---------|----------------|---------|---------|
| effnet-b0 | 0.66469| 0.75131| 0.86815 |    

### motor imagery classification.
| 模型 | physionet_acc | physionet_kappa | physionet_weighted_f1 | shu-mi_acc | shu-mi_kappa | shu-mi_weighted_f1 |
|------|---------------|-----------------|-----------------------|------------|--------------|--------------------|
| b0   | 0.64018       | 0.52011         | 0.64162               | 0.63499    | 0.71522      | 0.70266            |

### MENTAL DISORDER DIAGNOSIS

| model |    acc |   prauc |      rocauc |
|:------|-------:|--------:|-------:|
| b0    | 0.93833 | 0.98897 | 0.98695 | 

### IMAGINED SPEECH CLASSIFICATION 


| model | acc     | kappa   | f1      |
|-------|---------|---------|---------|
| b0    | 0.62667 | 0.53333 | 0.62595 |


### SLEEP STAGING
| model | acc     | kappa   | f1      |
|-------|---------|---------|---------|
| b0    | 0.76129 | 0.83855 | 0.83457 |

### Emotion Recognition

| model         | faced_acc  | faced_kappa | faced_weighted_f1 | seedv_acc | seedv_kappa | seedv_weighted_f1 |
|---------------|------------|-----------|-----------------|----------|-----------|-----------------|
| b0            | 0.55059    | 0.49360   | 0.55441         |     |      |            |
| convnext-tiny |   |      |            | 0.40219        | 0.25169         | 0.40166               |