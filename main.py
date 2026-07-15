import torch
import torchvision
import glob
import cv2
import numpy as np

def bgr2gray(img):
    w = torch.tensor([0.1140, 0.5870, 0.2989]).view(3, 1, 1) # across channel dim
    img = (w * img.permute(2, 0, 1)).sum(dim=0, keepdim=True)
    return img

def rgb2lab_l(img):
    r,g,b = img[0], img[1], img[2]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    l_channel  = torch.where(y > 0.008856, 116.0 * torch.pow(y, 1/3) - 16.0, 903.3 * y)
    return torch.clamp(l_channel, 0.0, 100.0) # CIELAB lightness range

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

def exposure_metrics(img):
    N = img[0].numel()
    L = rgb2lab_l(img)
    H = torch.histc(L, bins=100, min=0.0, max=100.0)
    
    N_s = H[0:2].sum()
    R_s = (N_s / N).item()

    N_h = H[98:100].sum()
    R_h = (N_h / N).item()
    
    Lvar = torch.std(L).item()
    
    return {"shadow clipping": R_s, "highlight clipping": R_h, "luminance variance": Lvar}

def SNR(img):
    N = img[0].numel()
    L = rgb2lab_l(img)
    P_signal = torch.sum(L**2) / N
    P_noise = torch.var(L)
    
    return 10 * torch.log10(P_signal / P_noise + 1e-8)

def edge_density(img, threshold=0.1):
    if img.max() > 1.0:
        img = img.float() / 255.0
    img = img.permute(2, 0, 1)
    L = rgb2lab_l(img) / 100.0
    L = L.unsqueeze(0).unsqueeze(0)
    smoothing = torch.Tensor([1, 2, 1])
    diff = torch.Tensor([1, 0, -1])
    kernel_x = torch.outer(smoothing, diff) / 4.0
    kernel_y = torch.outer(diff, smoothing) / 4.0
    
    kernel_x = kernel_x.view(1, 1, 3, 3)
    kernel_y = kernel_y.view(1, 1, 3, 3)
    
    G_x = torch.nn.functional.conv2d(L, kernel_x, padding=1)
    G_y = torch.nn.functional.conv2d(L, kernel_y, padding=1)
    G = torch.sqrt(G_x**2 + G_y**2)
    E = (G >= threshold).float()
    
    return E.mean().item()

pths = glob.glob('seed_data/img_*.jpg')
imgs = []
bimgs = []
egs = []
grains = []
blurs = []
snrs = []
exposurems = []
shadow_clips = []
highlight_clips = []
lum_vars = []
for pth in pths:
    img = cv2.imread(pth)
    img = torch.Tensor(img) / 255.0
    grains.append(graininess(img))
    blurs.append(blurriness(img))
    egs.append(edge_density(img))
    snrs.append(SNR(img))
    exp_data = exposure_metrics(img)
    shadow_clips.append(exp_data["shadow clipping"])
    highlight_clips.append(exp_data["highlight clipping"])
    lum_vars.append(exp_data["luminance variance"])
    imgs.append(img)

imgs = np.array(imgs)
fair_vec = np.stack((
    blurs, 
    grains, 
    egs, 
    snrs, 
    shadow_clips, 
    highlight_clips, 
    lum_vars
), axis=1)

# print(fair_vec)

good_imgs = fair_vec[:24, :]
bad_imgs = fair_vec[24:, :]

thresholds = {"blur": np.percentile(good_imgs[:, 0], 95),
              "grain": np.percentile(good_imgs[:, 1], 95),
              "edge": np.percentile(good_imgs[:, 2], 95),
              "snr": np.percentile(good_imgs[:, 3], 5),
              "shadow": np.percentile(good_imgs[:, 4], 95),
              "highlight": np.percentile(good_imgs[:, 5], 95),
              "lumvar": np.percentile(good_imgs[:, 6], 5)}

def reliability(fair_vec, thresholds):
    # compute expected confidence score (0.0 to 1.0) base on how well metrics respect thresholds
    penalties = []
    blur_ratio = fair_vec[0] / thresholds["blur"]
    penalties.append(min(blur_ratio - 1.0, 1.0) if blur_ratio > 1.0 else 0.0)
    
    grain_ratio = fair_vec[1] / thresholds["grain"]
    penalties.append(min(grain_ratio - 1.0, 1.0) if grain_ratio > 1.0 else 0.0)
    
    edge_ratio = fair_vec[2] / thresholds["edge"]
    penalties.append(min(edge_ratio - 1.0, 1.0) if edge_ratio > 1.0 else 0.0)
    
    snr_ratio = thresholds["snr"] / max(fair_vec[3], 1e-5)
    penalties.append(min(snr_ratio - 1.0, 1.0) if snr_ratio > 1.0 else 0.0)

    sh_ratio = fair_vec[4] / thresholds["shadow"]
    penalties.append(min(sh_ratio - 1.0, 1.0) if sh_ratio > 1.0 else 0.0)

    hi_ratio = fair_vec[5] / thresholds["highlight"]
    penalties.append(min(hi_ratio - 1.0, 1.0) if hi_ratio > 1.0 else 0.0)
    
    lv_ratio = thresholds["lumvar"] / max(fair_vec[6], 1e-5)
    penalties.append(min(lv_ratio - 1.0, 1.0) if lv_ratio > 1.0 else 0.0)
    
    avg_penalty = sum(penalties) / len(penalties)
    expected_conf = 0.95 - (avg_penalty * 0.45)
    return max(expected_conf, 0.50)
    
print(reliability(fair_vec=fair_vec[45], thresholds=thresholds))