import torch
import numpy as np
import pandas as pd
import cv2 # opencv
import glob
import matplotlib.pyplot as plt

pths = glob.glob("lemon_healthy/*.JPG")
himgs = []
for pth in pths:
    himgs.append(cv2.cvtColor(cv2.imread(pth), cv2.COLOR_BGR2RGB).reshape(1440, 1080, 3))
himgs = np.stack(himgs)
print(himgs.shape)

hlbpths = glob.glob("lemon_HLB/*.JPG")
hlbimgs = []
for pth in hlbpths:
    hlbimgs.append(cv2.cvtColor(cv2.imread(pth), cv2.COLOR_BGR2LAB).reshape(1440, 1080, 3))
hlbimgs = np.stack(hlbimgs)
print(hlbimgs.shape)

plt.imshow(hlbimgs[100])
plt.show()

