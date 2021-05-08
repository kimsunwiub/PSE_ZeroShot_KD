import numpy as np
import torch
import torch.nn as nn
   
from data import mix_signals_batch, prep_sig_ml

eps = 1e-6

def stft(signal, fft_size, hop_size):
    window = torch.hann_window(fft_size, device=signal.device)
    S = torch.stft(signal, n_fft=fft_size, hop_length=hop_size, window=window)#, return_complex=False)
    return S

def get_magnitude(S):
    S_mag = torch.sqrt(S[..., 0] ** 2 + S[..., 1] ** 2 + 1e-20)
    return S_mag

def apply_mask(spectrogram, mask, device):
    assert (spectrogram[...,0].shape == mask.shape)
    spectrogram2 = torch.zeros(spectrogram.shape)
    spectrogram2[..., 0] = spectrogram[..., 0] * mask
    spectrogram2[..., 1] = spectrogram[..., 1] * mask
    return spectrogram2.to(device)

def istft(spectrogram, fft_size, hop_size):
    window = torch.hann_window(fft_size, device=spectrogram.device)
    y = torch.istft(spectrogram, n_fft=fft_size, hop_length=hop_size, window=window)
    return y

# Train Utils
def calculate_sdr(source_signal, estimated_signal, offset=None, scale_invariant=False):
    """
    Imported from: https://github.com/IU-SAIGE/sparse_mle
    """
    s = source_signal.clone()
    y = estimated_signal.clone()

    # add a batch axis if non-existant
    if len(s.shape) != 2:
        s = s.unsqueeze(0)
        y = y.unsqueeze(0)

    # truncate all signals in the batch to match the minimum-length
    min_length = min(s.shape[-1], y.shape[-1])
    s = s[..., :min_length]
    y = y[..., :min_length]

    if scale_invariant:
        alpha = s.mm(y.T).diag()
        alpha /= ((s ** 2).sum(dim=1) + eps)
        alpha = alpha.unsqueeze(1)  # to allow broadcasting
    else:
        alpha = 1

    e_target = s * alpha
    e_res = e_target - y

    numerator = (e_target ** 2).sum(dim=1)
    denominator = (e_res ** 2).sum(dim=1) + eps
    sdr = 10 * torch.log10((numerator / denominator) + eps)

    # if `offset` is non-zero, this function returns the relative SDR
    # improvement for each signal in the batch
    if offset is not None:
        sdr -= offset

    return sdr

def calculate_sisdr(source_signal, estimated_signal, offset=None):
    """
    Imported from: https://github.com/IU-SAIGE/sparse_mle
    """
    return calculate_sdr(source_signal, estimated_signal, offset, True)

def loss_sdr(source_signal, estimated_signal, offset=None):
    """
    Imported from: https://github.com/IU-SAIGE/sparse_mle
    """
    return -1.*torch.mean(calculate_sdr(source_signal, estimated_signal, offset))

def loss_sisdr(source_signal, estimated_signal, offset=None):
    """
    Imported from: https://github.com/IU-SAIGE/sparse_mle
    """
    return -1.*torch.mean(calculate_sisdr(source_signal, estimated_signal, offset))

def denoise_signal(args, mix_batch, G_model):
    """
    Return predicted clean speech.
    
    mix_batch and G_model: Located on GPU.
    """
    X = stft(mix_batch, args.fft_size, args.hop_size)
    X_mag = get_magnitude(X).permute(0,2,1)
    mask_pred = G_model(X_mag).permute(0,2,1)
    mask_pred = mask_pred.unsqueeze(3).repeat(1,1,1,2)
    X_est = X * mask_pred
    est_batch = istft(X_est, args.fft_size, args.hop_size)
    return est_batch

def denoise_signal_ctn(args, mix_batch, G_model):
    return G_model(mix_batch).squeeze(1)

def run_iter(args, tot_s, tot_x, student_model, ori_student_model, teacher_model, student_optimizer=None, trtt=None):
    stu_res = []
    tea_res = []
    ori_res = []
    loss_res = []
    for idx in range(0,len(tot_s),args.batch_size):
        speech_batch = tot_s[idx:idx+args.batch_size].to(args.device)
        mix_batch = tot_x[idx:idx+args.batch_size]
        
        stu_e = denoise_signal(args, mix_batch.to(args.device), student_model)
        if args.ctn_tea:
            if trtt == 'test': # Compute signals individually since test signals are 10s long and overloads GPU memory.
                tea_e = []
                for x in mix_batch:
                    x = x[None,:]
                    tea_e_i = denoise_signal_ctn(args, x.to(args.device), teacher_model).squeeze(1)
                    tea_e_i = tea_e_i.detach().cpu()
                    _, tea_e_i, _ = prep_sig_ml(tea_e_i, stu_e)
                    tea_e.append(tea_e_i)
                tea_e = torch.stack(tea_e).squeeze(1)
            else:
                tea_e = denoise_signal_ctn(args, mix_batch.to(args.device), teacher_model)
                _, tea_e, _ = prep_sig_ml(tea_e, stu_e)
            tea_e = tea_e.detach().cpu().to(args.device)
        else:
            tea_e = denoise_signal(args, mix_batch.to(args.device), teacher_model)
        mix_batch = mix_batch.to(args.device)
        ori_e = denoise_signal(args, mix_batch, ori_student_model)

        # Truncate to same lengths
        _, s, _ = prep_sig_ml(speech_batch, stu_e)
        _, x, _ = prep_sig_ml(mix_batch, stu_e)

        # Standardize
        s = s/(s.std(1)[:,None] + eps)
        x = x/(x.std(1)[:,None] + eps)
        stu_e = stu_e/(stu_e.std(1)[:,None] + eps)
        ori_e = ori_e/(ori_e.std(1)[:,None] + eps)
        tea_e = tea_e/(tea_e.std(1)[:,None] + eps)

        stu_sdr = float(calculate_sisdr(s, stu_e).mean())
        tea_sdr = float(calculate_sisdr(s, tea_e).mean())
        ori_sdr = float(calculate_sisdr(s, ori_e).mean())
        
        stu_res.append(stu_sdr)
        tea_res.append(tea_sdr)
        ori_res.append(ori_sdr)

        if args.sisdr_loss:
            mix_offset = calculate_sisdr(tea_e, x) # Offset is computed with s_T as ground-truth. 
            loss_i = loss_sisdr(tea_e, stu_e, offset=mix_offset) 
        else:
            loss_i = loss_fn(tea_e, stu_e)
            
        if student_optimizer:
            student_optimizer.zero_grad()
            loss_i.backward()
            student_optimizer.step()
            
        loss_res.append(float(loss_i))
        del loss_i

    return np.mean(stu_res), np.mean(tea_res), np.mean(ori_res), np.mean(loss_res)       

def run_se_ctn(args, model, speech_dataloader, noise_dataloader, is_train=True, optimizer=None):
    total_loss = np.zeros(len(speech_dataloader))

    noise_iter = iter(noise_dataloader)
    for batch_idx, speech_batch in enumerate(speech_dataloader):
        try:
            noise_batch = next(noise_iter)
        except StopIteration:
            noise_iter = iter(noise_dataloader)
            noise_batch = next(noise_iter)

        snr_db_batch = np.random.uniform(
            low=min(args.snr_ranges), high=max(args.snr_ranges), size=args.batch_size).astype(
            np.float32)

        mix_batch = mix_signals_batch(speech_batch, noise_batch, snr_db_batch).to(args.device)
        est_batch = model(mix_batch).squeeze(1)

        if is_train:
            optimizer.zero_grad()

        speech_batch = speech_batch.to(args.device)
        actual_sisdr = calculate_sisdr(speech_batch, mix_batch)
        loss_i = loss_sisdr(speech_batch, est_batch, actual_sisdr)
        total_loss[batch_idx] = float(loss_i)

        if is_train:
            loss_i.backward()
            optimizer.step()
        
        del loss_i
        torch.cuda.empty_cache()

        if (batch_idx % args.print_every) == 0: 
            print ("Batch {}. Loss: {:.3f} SI-SDRi".format(
                    batch_idx, total_loss[:batch_idx+1].mean())) # logging.info

    return total_loss[:batch_idx+1].mean()

def run_se(args, model, speech_dataloader, noise_dataloader, is_train=True, optimizer=None):
        total_loss = np.zeros(len(speech_dataloader))
        
        noise_iter = iter(noise_dataloader)
        for batch_idx, speech_batch in enumerate(speech_dataloader):
            try:
                noise_batch = next(noise_iter)
            except StopIteration:
                noise_iter = iter(noise_dataloader)
                noise_batch = next(noise_iter)

            snr_db_batch = np.random.uniform(
                low=min(args.snr_ranges), high=max(args.snr_ranges), size=args.batch_size).astype(
                np.float32)

            mix_batch = mix_signals_batch(speech_batch, noise_batch, snr_db_batch).to(args.device)
            X = stft(mix_batch, args.fft_size, args.hop_size)
            X_mag = get_magnitude(X)
            X_mag = X_mag.permute(0,2,1)
            mask_pred = model(X_mag)
            mask_pred = mask_pred.permute(0,2,1)
            mask_pred = mask_pred.unsqueeze(3).repeat(1,1,1,2)
            X_est = X * mask_pred
            est_batch = istft(X_est, args.fft_size, args.hop_size)

            if is_train:
                optimizer.zero_grad()

            speech_batch = speech_batch.to(args.device)
            actual_sisdr = calculate_sisdr(speech_batch, mix_batch)
            loss_i = loss_sisdr(speech_batch, est_batch, actual_sisdr)
            total_loss[batch_idx] = loss_i

            if is_train:
                loss_i.backward()
                optimizer.step()

            if (batch_idx % args.print_every) == 0: 
                print ("Batch {}. Loss: {:.3f} SI-SDRi".format(
                        batch_idx, total_loss[:batch_idx+1].mean())) # logging.info
        
        return total_loss[:batch_idx+1].mean()
    