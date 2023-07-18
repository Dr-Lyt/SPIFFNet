

from option import args
import data
import model
import utils
import loss
import trainer
from data.ucmerced import UCMercedDataset
from torch.utils.data import DataLoader
import torch
import data.common as common
import cv2
import glob
import os
import numpy as np
if __name__ == '__main__':
    device = torch.device('cpu' if args.cpu else 'cuda')

    args.dir_data = '../UCMerced-dataset/test/LR_x4'
    args.model = 'SPIFFNET'
    args.pre_train = '../model_best.pt'
    args.dir_out = '..'

    args.swinir_depths = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    args.swinir_num_heads = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    args.swinir_window_size = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16]
    args.swinir_embed_dim = 64

    args.resume = 0
    args.print_model = False
    args.test_only = True
    checkpoint = utils.checkpoint(args)
    sr_model = model.Model(args, checkpoint)

    sr_model.eval()
    save_img = False
    root_dir = '../UCMerced-dataset/test'
    dir_hr = os.path.join(root_dir, 'HR_x' + str(args.scale[0]))
    dir_lr = os.path.join(root_dir, 'LR_x' + str(args.scale[0]))
    list_hr = glob.glob(os.path.join(dir_hr, '*.tif'))
    list_lr = []
    crop_border = args.scale[0]
    eval_acc = 0
    eval_pnsr_acc = 0
    eval_ssim_acc = 0
    img_num = 0
    for i in range(len(list_hr)):
        filename = os.path.split(list_hr[i])[-1]
        lr_path=os.path.join(dir_lr, filename)
        hr_np = cv2.imread(list_hr[i], cv2.IMREAD_COLOR)
        hr_np = cv2.cvtColor(hr_np, cv2.COLOR_BGR2RGB)
        lr_np = cv2.imread(lr_path, cv2.IMREAD_COLOR)
        lr_np = cv2.cvtColor(lr_np, cv2.COLOR_BGR2RGB)
        lr = common.np2Tensor([lr_np], args.rgb_range)[0].unsqueeze(0)
        hr = common.np2Tensor([hr_np], args.rgb_range)[0].unsqueeze(0)
        lr = lr.to(device)
        hr = hr.to(device)
        sr = sr_model(lr)
        sr = utils.quantize(sr, args.rgb_range)
        save_list = [sr]
        sr = utils.torch_to_np(sr)
        hr = utils.torch_to_np(hr)
        ####保存图像
        if save_img:
            final_sr = sr * 255.
            final_sr = final_sr.astype(np.uint8)
            img = cv2.cvtColor(final_sr[0, ...], cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(args.dir_out, filename + '.tif'), img)
        # crop borders
        if crop_border == 0:
            cropped_hr = hr
            cropped_sr = sr
        else:
            cropped_hr = hr[:, crop_border:-crop_border, crop_border:-crop_border, :]
            cropped_sr = sr[:, crop_border:-crop_border, crop_border:-crop_border, :]
        sig_pnsr=utils.calculate_psnr(cropped_sr, cropped_hr, args.rgb_range)
        eval_pnsr_acc += sig_pnsr
        if args.rgb_range == 1:
            sig_ssim=utils.calculate_batch_ssim(cropped_sr * 255, cropped_hr * 255)
            eval_ssim_acc += sig_ssim
        else:
            sig_ssim = utils.calculate_batch_ssim(cropped_sr * 255, cropped_hr * 255)
            eval_ssim_acc += sig_ssim

        img_num += sr.shape[0]

        ssim_acc = eval_ssim_acc / img_num
        pnsr_acc = eval_pnsr_acc / img_num
        log = '[{} {}x{}] {:3d} {}\t{}:{:.6f} {}:{:.6f}'.format(
            args.model,
            args.dataset,
            crop_border,
            i + 1,
            filename,
            'ssim',
            sig_ssim,
            'pnsr',
            sig_pnsr
        )
        print(log)
    print('Average: PSNR: {:.6f} dB, SSIM: {:.6f}'.format(
        pnsr_acc,
        ssim_acc))