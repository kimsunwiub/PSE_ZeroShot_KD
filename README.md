# Personalized Speech Enhanacement

About: We propose a novel personalized speech enhancement method to adapt a compact denoising model to the test-time specificity. Our goal in this test-time adaptation is to utilize no clean speech target of the test speaker, thus fulfilling the requirement for zero-shot learning. To complement the lack of clean utterance, we employ the knowledge distillation framework. Instead of the missing clean utterance target, we distill the more advanced denoising results from an overly large teacher model, and use it as the pseudo target to train the small student model. 

This material is based upon work supported by the National Science Foundation under Grant No. 2046963.

Paper: Test-Time Adaptation Toward Personalized Speech Enhancement:Zero-Shot Learning with Knowledge Distillation. Link: TBD

More updates/clean ups to be done, pleqse contact authors for immediate questions. 

## Datasets used in this repository
* LibriSpeech
* MUSA


 

### Repository structure

#### pretraining_generalist.py
* Pre-training speech enhanacement models

```
python pretrain_generalist.py --device 7 --learning_rate 1e-4 -r 2 -g 32 --is_save
```

#### pse_0shot_kd.py
* Test-time personalized speech enhanacement

```
python pse_0shot_kd.py -d 7 -l 5e-4 --sisdr_loss --is_save
```
