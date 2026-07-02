import torch
import glob
import cv2
import numpy as np

pths = glob.glob('seed_data/img_*.jpg')
imgs = []
bimgs = []
for pth in pths:
    img = cv2.imread(pth, cv2.IMREAD_GRAYSCALE)
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    lvar = laplacian.var()
    if lvar < 150:
        bimgs.append(img)
    if lvar > 150:
        imgs.append(img)
        
bimgs = torch.Tensor(np.stack(bimgs))
imgs = torch.Tensor(np.stack(imgs))
print(imgs.shape, bimgs.shape)