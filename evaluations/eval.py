import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

gt_dir = 'F:\\NeRD-Rain-main\Datasets\Rain200L\\test\\target'
pred_dir = 'F:\\NeRD-Rain-main\\results\Rain200L_all'

psnr_list = []
ssim_list = []

for name in os.listdir(gt_dir):
    gt_path = os.path.join(gt_dir, name)
    pred_path = os.path.join(pred_dir, name)

    if not os.path.exists(pred_path):
        continue

    gt = cv2.imread(gt_path)
    pred = cv2.imread(pred_path)

    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)

    psnr = peak_signal_noise_ratio(gt, pred, data_range=255)
    ssim = structural_similarity(gt, pred, channel_axis=2, data_range=255)

    psnr_list.append(psnr)
    ssim_list.append(ssim)

print('Average PSNR:', np.mean(psnr_list))
print('Average SSIM:', np.mean(ssim_list))
