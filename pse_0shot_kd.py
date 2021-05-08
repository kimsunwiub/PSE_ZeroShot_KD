import os
from argparse import ArgumentParser
import logging

import torch
torch.set_num_threads(1)

import torch.nn as nn

from data import *
from models import *
from se_kd_utils import *
from utils import *

def parse_arguments():
    parser = ArgumentParser()
        
    parser.add_argument("-d", "--device", type=int)
    parser.add_argument("-c", "--seed", type=int, default=-1)
    parser.add_argument("-l", "--learning_rate", type=float, default=1e-5)
    
    parser.add_argument("-s", "--student_hidden_size", type=int, default=-1)
    parser.add_argument("-e", "--teacher_num_layers", type=int, default=-1)
    parser.add_argument("-t", "--teacher_hidden_size", type=int, default=-1)    
    parser.add_argument("--load_student_model", type=str, default=None)
    parser.add_argument('--load_teacher_model', type=str, default=None)
    parser.add_argument("--snr_ranges", nargs='+', type=int, default=[-6])

    parser.add_argument("--batch_size", type=int, default=15)
    parser.add_argument("--tot_epoch", type=int, default=200)
    parser.add_argument("--student_num_layers", type=int, default=2)
    
    parser.add_argument("--data_dir", type=str, default="/home/kimsunw/data/")
    parser.add_argument("--save_dir", type=str, default="/home/kimsunw/workspace/pse/kd_results/")
    
    parser.add_argument("--load_student_data", type=str, default=None)
    parser.add_argument('--load_teacher_data', type=str, default=None)
    
    parser.add_argument("--print_every", type=int, default=5)
    parser.add_argument("--validate_every", type=int, default=2)
    
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--duration", type=int, default=1)
    parser.add_argument("--fft_size", type=int, default=1024)
    parser.add_argument("--hop_size", type=int, default=256)
    
    parser.add_argument('--is_save', action='store_true')
    parser.add_argument('--ctn_tea', action='store_true')
    parser.add_argument('--sisdr_loss', action='store_true')
        
    return parser.parse_args()
      
args = parse_arguments()
logging.getLogger().setLevel(logging.INFO)
args.is_train_kd = True
args.is_train_disc = False
args.stft_features = int(args.fft_size//2+1)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= str(args.device)
args.device = 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

### Model
m1k_dir = "new_models_results/Gs/expr04221825_SE_G2x1024_lr1e-04_bs100_ctnFalsesm-1_nfrms16000_GPU4/"
m256_dir = "new_models_results/Gs/expr04221825_SE_G2x256_lr1e-04_bs100_ctnFalsesm-1_nfrms16000_GPU6/"
m32_dir = "new_models_results/Gs/expr04221825_SE_G2x32_lr1e-04_bs100_ctnFalsesm-1_nfrms16000_GPU6/"
m512_dir = "new_models_results/Gs/expr04221825_SE_G2x512_lr1e-04_bs100_ctnFalsesm-1_nfrms16000_GPU5/"
m3x1k_dir = "new_models_results/Gs/expr04221825_SE_G3x1024_lr1e-04_bs100_ctnFalsesm-1_nfrms16000_GPU3/"
m128_dir = "new_models_results/Gs/expr04221826_SE_G2x128_lr1e-04_bs100_ctnFalsesm-1_nfrms16000_GPU7/"
m64_dir = "new_models_results/Gs/expr04221826_SE_G2x64_lr1e-04_bs100_ctnFalsesm-1_nfrms16000_GPU7/"
ctn_dir = "new_models_results/Gs/expr04221819_SE_G3x1024_lr5e-04_bs20_ctnTruesm-1_nfrms16000_GPU1/"

dirs = [
    m32_dir,
    m64_dir,
    m128_dir,
    m256_dir,
    m512_dir,
    m1k_dir
]

hus = [32, 64, 128, 256, 512, 1024, 1024]
hls = [2, 2, 2, 2, 2, 2, 3]

snr_ranges_all = [-5,0,5,10]
loss_fn = nn.MSELoss()
for seed in range(40):
    args.seed = seed
    print("Running for seed ", args.seed)
    
    # Load Teacher Model
    if args.ctn_tea:
        from asteroid.models import ConvTasNet
        teacher_model = ConvTasNet(n_src=1)
        teacher_model.load_state_dict(torch.load("{}/Dmodel_best.pt".format(ctn_dir)))
        teacher_model = teacher_model.to(args.device)
    else:
        args.teacher_hidden_size = hus[-1]
        args.teacher_num_layers = hls[-1]
        args.load_teacher_model = m3x1k_dir + 'Dmodel_best.pt'
        teacher_model = SpeechEnhancementModel(
            args.teacher_hidden_size, args.teacher_num_layers, args.stft_features)
        load_model(teacher_model, args.load_teacher_model)
        teacher_model = teacher_model.to(args.device)
        
    for i in range(len(dirs)):
        # Load baseline original student model
        args.student_hidden_size = hus[i]
        args.student_num_layers = hls[i]
        ori_student_model = SpeechEnhancementModel(
            args.student_hidden_size, args.student_num_layers, args.stft_features).to(args.device)
        load_model(ori_student_model, dirs[i] + 'Dmodel_best.pt')

        for snr_ranges in snr_ranges_all:
            args.snr_ranges = [snr_ranges]
            output_directory = setup_expr(args)
            
            tot_s, tot_n = init_pers_set(args)
            tr_s, tr_n, va_s, va_n, te_s, te_n = mixup(args, tot_s, tot_n)
            
            va_x = mix_signals_batch(va_s, va_n, args.snr_ranges)
            te_x = mix_signals_batch(te_s, te_n, args.snr_ranges)

            # Load student model for personalization
            student_model = SpeechEnhancementModel(
                args.student_hidden_size, args.student_num_layers, args.stft_features).to(args.device)
            load_model(student_model, dirs[i] + 'Dmodel_best.pt')
            student_optimizer = torch.optim.Adam(params=student_model.parameters(),lr=args.learning_rate)
            
            tr_stu_sisdr = []
            tr_tea_sisdr = []
            tr_mix_sisdr = []
            tr_ori_sisdr = []
            tr_losses = []
            
            va_stu_sisdr = []
            va_tea_sisdr = []
            va_mix_sisdr = []
            va_ori_sisdr = []
            va_losses = []
            
            te_stu_sisdr = []
            te_tea_sisdr = []
            te_mix_sisdr = []
            te_ori_sisdr = []
            te_losses = []

            for ep in range(args.tot_epoch):
                
                tr_s = shuffle_set(tr_s)
                tr_x = mix_signals_batch(tr_s, shuffle_set(tr_n)[:len(tr_s)], args.snr_ranges)
                tr_len = len(tr_x)
                tr_s = tr_s[:tr_len-(tr_len%args.batch_size)]
                tr_x = tr_x[:tr_len-(tr_len%args.batch_size)]
                
                stu_sdr, tea_sdr, ori_sdr, loss_i = run_iter(
                    args, tr_s, tr_x, student_model, ori_student_model, teacher_model, 
                    student_optimizer=student_optimizer)

                tr_stu_sisdr.append(stu_sdr)
                tr_tea_sisdr.append(tea_sdr)
                tr_ori_sisdr.append(ori_sdr)
                tr_losses.append(loss_i)
                
                stu_sdr, tea_sdr, ori_sdr, loss_i = run_iter(
                    args, va_s, va_x, student_model, ori_student_model, teacher_model)
                
                va_stu_sisdr.append(stu_sdr)
                va_tea_sisdr.append(tea_sdr)
                va_ori_sisdr.append(ori_sdr)
                va_losses.append(loss_i)
                
                stu_sdr, tea_sdr, ori_sdr, loss_i = run_iter(
                    args, te_s, te_x, student_model, ori_student_model, teacher_model, trtt='test')
                
                te_stu_sisdr.append(stu_sdr)
                te_tea_sisdr.append(tea_sdr)
                te_ori_sisdr.append(ori_sdr)
                te_losses.append(loss_i)
                
            logging.info("Epoch {} Training. Student: {:.2f} | Orig: {:.2f} | Teacher: {:.2f} | Loss: {:.2f}".format(
                ep, 
                tr_stu_sisdr[-1], 
                tr_ori_sisdr[-1], 
                tr_tea_sisdr[-1], 
                tr_losses[-1]
            ))

            logging.info("Epoch {} Validation. Student: {:.2f} | Orig: {:.2f} | Teacher: {:.2f} | Loss: {:.2f}".format(
                ep, 
                va_stu_sisdr[-1], 
                va_ori_sisdr[-1], 
                va_tea_sisdr[-1], 
                va_losses[-1]
            ))

            logging.info("Epoch {} Testing. Student: {:.2f} | Orig: {:.2f} | Teacher: {:.2f} | Loss: {:.2f}".format(
                ep, 
                te_stu_sisdr[-1], 
                te_ori_sisdr[-1], 
                te_tea_sisdr[-1], 
                te_losses[-1]
            ))

            seed_dict = {
                "tr_stu_sisdr" : tr_stu_sisdr,
                "tr_ori_sisdr" : tr_ori_sisdr,
                "tr_tea_sisdr" : tr_tea_sisdr,
                "tr_losses" : tr_losses,
                
                "va_stu_sisdr" : va_stu_sisdr,
                "va_ori_sisdr" : va_ori_sisdr,
                "va_tea_sisdr" : va_tea_sisdr,
                "va_losses" : va_losses,

                "te_stu_sisdr" : te_stu_sisdr,
                "te_ori_sisdr" : te_ori_sisdr,
                "te_tea_sisdr" : te_tea_sisdr,
                "te_losses" : te_losses,
                
                "epoch": 666,
            }

            if args.is_save:
                save_model(student_model, output_directory, seed_dict, is_last=True)
            
logging.info("Finished KD")