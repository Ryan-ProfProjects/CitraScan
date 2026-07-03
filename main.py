import torch
import torchvision
import glob
import cv2
import numpy as np

def bgr2gray(img):
    w = torch.tensor([0.1140, 0.5870, 0.2989]).view(3, 1, 1) # across channel dim
    img = (w * img.permute(2, 0, 1)).sum(dim=0, keepdim=True)
    return img

def blurriness(img):
    # bgr to grayscale
    img = bgr2gray(img)
    # conv
    lkernel = torch.Tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3)
    lvar = torch.var(torch.nn.functional.conv2d(img.unsqueeze(0), lkernel, padding=1))
    N = img.numel() # resolution
    sidx = lvar/np.sqrt(N)
    return sidx.item()
    
def graininess(img):
    blur = torchvision.transforms.GaussianBlur(kernel_size=3, sigma=0.5)
    blurimg = blur(img)
    noise = img - blurimg
    noise = bgr2gray(noise)
    # conv
    lkernel = torch.Tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3)
    return torch.var(torch.nn.functional.conv2d(noise.unsqueeze(0), lkernel, padding=1)).item()
    
def EI(img):
    img = bgr2gray(img)
    H = torch.histogram(img, bins=torch.arange(0, 257, dtype=torch.float32))
    nclipped = H[0][0].item() + H[0][255].item()
    N = img.numel()
    rclip = nclipped / N
    probs = H[0] / N
    
    # fix nan issue
    active_probs = probs[probs > 0]
    E = -torch.sum(active_probs * (torch.log(active_probs) / np.log(2))).item() # torch.log is base e
    return (1-rclip) * E * 0.125

pths = glob.glob('seed_data/img_*.jpg')
imgs = []
bimgs = []
EIs = []
for pth in pths:
    img = cv2.imread(pth)
    EIs.append(EI(torch.Tensor(img)))

eis = np.array(EIs)
# min-max scaling to exaggerate relative differences
eis = (eis - eis.min()) / (eis.max() - eis.min())
eis = np.clip(eis, 0.0, 1.0)
print(eis)