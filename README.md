# EEG-Vision

+ A solution from vision perspective to solve eeg problem

Although numerous methods have been proposed to address EEG-related problems,
including the 'top' unsupervised method,  **Labram, Cbramod, EEG-Dino[1,2,3]**.
I believe that most existing works(basically all) suffer from **overclaiming** issues and
fail to capture the essential characteristics of EEG data. Based on this
observation, I developed this project that innovatively employs vision models
to tackle EEG problems.

This project builds upon the Cbramod framework and
conducts comprehensive comparative experiments across the 12 downstream tasks
claimed in the Cbramod paper, obtaining corresponding experimental results.
The results will tell the story. 

**Please beat existing baselines first before claiming SOTA**

## 🚀 Result

Here it goes,

### 1.1 chb-mit

| model     | acc     | prauc   | rocuac  |
|-----------|---------|---------|---------|
| effnet-b0 | 0.70594 | 0.42923 | 0.89768 | 

### 1.2 tuab

| model            | acc     | prauc   | rocauc  |
|------------------|---------|---------|---------|
| effnet-b0        | 0.82049 | 0.89364 | 0.89177 |    
| convnextv2_small | 0.83136 | 0.91319 | 0.90551 |     

### 1.3 TUEV

| model     | acc     | prauc   | rocauc  |
|-----------|---------|---------|---------|
| effnet-b0 | 0.66469 | 0.75131 | 0.86815 |    

### 1.4 motor imagery classification.

| 模型 | physionet  |        |               | shu-mi |        |               |
| ---- | ----------------- | ------ | ------------- | ------------- | ------ | ------------- |
|      | acc               | kappa  | weighted_f1   | acc           | kappa  | weighted_f1   |
| b0   | 0.64018           | 0.52011| 0.64162       | 0.63499       | 0.71522| 0.70266       |
### 1.5 MENTAL DISORDER DIAGNOSIS

| model |     acc |   prauc |  rocauc |
|:------|--------:|--------:|--------:|
| b0    | 0.93833 | 0.98897 | 0.98695 | 

### 1.6 IMAGINED SPEECH CLASSIFICATION

| model | acc     | kappa   | f1      |
|-------|---------|---------|---------|
| b0    | 0.62667 | 0.53333 | 0.62595 |

### 1.7 SLEEP STAGING

| model | acc     | kappa   | f1      |
|-------|---------|---------|---------|
| b0    | 0.76129 | 0.83855 | 0.83457 |

### 1.8 Emotion Recognition
|  | faced   |         |             | seed-v  |         |             |
| ---- |---------|---------|-------------|---------|---------|-------------|
|      | acc     | kappa   | weighted_f1 | acc     | kappa   | weighted_f1 |
| b0            | 0.55059 | 0.49360 | 0.55441     | --      | --      | --          |
| convnext-tiny | --      | --      | --          | 0.40219 | 0.25169 | 0.40166     |

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
or
bash train_tuev.sh
or 
bash train_speech.sh ...
```

We have released a pretrained checkpoint on [Hugginface🤗](https://huggingface.co/weighting666/CBraMod).

## ref

1. Wang, J., Zhao, S., Luo, Z., Zhou, Y., Jiang, H., Li, S., ... & Pan, G. (2024). Cbramod: A criss-cross brain
   foundation model for eeg decoding. arXiv preprint arXiv:2412.07236.
2. Wang, X., Liu, X., Liu, X., Si, Q., Xu, Z., Li, Y., & Zhen, X. (2025, September). Eeg-dino: Learning eeg foundation
   models via hierarchical self-distillation. In International Conference on Medical Image Computing and
   Computer-Assisted Intervention (pp. 196-205). Cham: Springer Nature Switzerland.
3. Jiang, W. B., Zhao, L. M., & Lu, B. L. (2024). Large brain model for learning generic representations with tremendous
   EEG data in BCI. arXiv preprint arXiv:2405.18765.