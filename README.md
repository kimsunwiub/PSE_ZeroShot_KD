# Personalized Speech Enhancement

This repository provides a script for training and personalizing speech enhancement (PSE) models to test-time environments.  The original PSE framework was introduced in [1]. 

## Datasets used in this repository
* LibriSpeech (https://www.openslr.org/12)
* MUSAN (https://www.openslr.org/17/)
* WHAM! (https://drive.google.com/file/d/1PEDhL0gKFfe70cwo6fm33x8XGVMxgfS3/view?usp=sharing) (Link to original work: https://wham.whisper.ai/)

Download the datasets into a location (e.g., data/). LibriSpeech and MUSAN can be downloaded in the provided link above. WHAM! corpus can also be downloaded in the provided original link, but the files for this project were organized manually with respect to each recorded location according to the provided metadata. Please download the formatted WHAM! corpus through the google drive link. 

## Usage
Pre-training speech enhanacement models can be done by running the ```pretraining_generalist.py``` script. For example,

```
python pretrain_generalist.py --device 0 --learning_rate 1e-4 -r 2 -g 32 --data_dir data/ --save_dir pretrained_models/ --is_save
```
to train a ```2x32``` model on GPU card ```0``` using ```1e-4``` learning rate. The models are saved to ```---save_dir``` value, which is ```pretrained_models/``` by default. 

Once the models have been prepared through pretraining, personalization can be done through the ```pse_kd.py``` script. For example, if the pretraining files have been saved to ```new_models_results/Gs/``` (the value ```--save_dir``` was set to during pretraining):

```
 python pse_kd.py --device 7 -student_num_layers 2 -student_hidden_size 32 -teacher_num_layers 3 -teacher_hidden_size 1024 -l 5e-4 --sisdr_loss --snr_ranges 0 --student_dir new_models_results/Gs/expr04221825_SE_G2x32_lr1e-04_bs100_ctnFalsesm-1_nfrms16000_GPU6/Dmodel_best.pt --teacher_dir new_models_results/Gs/expr04221825_SE_G3x1024_lr1e-04_bs100_ctnFalsesm-1_nfrms16000_GPU3/Dmodel_best.pt --data_dir ~/data/ --is_save
```
to load a pretrained ```2x32``` student model and pretrained ```3x1024``` teacher model. In the above example, learning rate of ```5e-4```, SISDRi [2] as the loss function, and mixing SNR is set to 0dB. At the end of the script, the results will appear as: 
```
INFO:root:Epoch 199 Training. Student: 6.74 | Orig: 3.52 | Teacher: 7.14 | Loss: -11.36
INFO:root:Epoch 199 Validation. Student: 7.08 | Orig: 4.58 | Teacher: 7.89 | Loss: -11.50
INFO:root:Epoch 199 Testing. Student: 7.34 | Orig: 4.51 | Teacher: 8.61 | Loss: -11.88
```
showing the performance of the personalized student, baseline pretrained student (unupdated), and teacher model in terms of SI-SDRi. 

### Notes
- For both pretraining and personalization, make sure the datasets (LibriSpeech, MUSAN, and formatted WHAM!) are placed in the data/ directory. 
- For personalization, use different ```seed``` values to vary the test-time environment. Upon running the script, the speaker and noise location will be logged. For example, 
```
Session Spkr:  2094
Noise Class:  Tomatina
```
- Since the test-time environment is defined by the noise recording location, download the formatted WHAM! corpus as mentioned in the earlier section. 
- For both pretraining and personalization, there is an additional option to use a ConvTasNet [3] model which can be set by adding the ```--ctn_tea``` option. 
- To skip the pretraining step, please find pretrained models available at https://drive.google.com/file/d/1QmuPuK5xoiNPKBx7avqdLDdXpizTDit5/view?usp=sharing
- To replicate the experiment in [1], use seed values from [0,40] and SNR ranges [-5,0,5,10]. 


### References
1. Kim, Sunwoo, and Minje Kim. "Test-time adaptation toward personalized speech enhancement: Zero-shot learning with knowledge distillation." 2021 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA). IEEE, 2021. (https://ieeexplore.ieee.org/abstract/document/9632771).
2. Le Roux, Jonathan, et al. "SDR–half-baked or well done?." ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2019.
3. Luo, Yi, and Nima Mesgarani. "Conv-tasnet: Surpassing ideal time–frequency magnitude masking for speech separation." IEEE/ACM transactions on audio, speech, and language processing 27.8 (2019): 1256-1266.
