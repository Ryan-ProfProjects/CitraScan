import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import os

def bgr2gray(img):
    w = torch.tensor([0.1140, 0.5870, 0.2989]).view(3, 1, 1) # across channel dim
    img = (w * img).sum(dim=0, keepdim=True)
    return img

def rgb2lab_l(img):
    r,g,b = img[0], img[1], img[2]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    l_channel  = torch.where(y > 0.008856, 116.0 * torch.pow(y, 1/3) - 16.0, 903.3 * y)
    return torch.clamp(l_channel, 0.0, 100.0)

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

def to_relvec(img):
    if img.max() > 1.0:
        img = img.float() / 255.0
        
    em = exposure_metrics(img)
    fair_vec = np.array((
        blurriness(img), 
        graininess(img), 
        edge_density(img), 
        SNR(img), 
        em["shadow clipping"], 
        em["highlight clipping"], 
        em["luminance variance"]
    ))

    return fair_vec

def score(fair_vec, reference_good):
    """
    normalizes reliability vector into reliability score between 0 and 1
    """

    blur, grain, edge, snr, shadow, highlight, lumvar = fair_vec
    
    ref_blur, ref_grain, _, ref_snr, _, _, ref_lumvar = reference_good

    blur_penalty = np.clip((blur - ref_blur) / (ref_blur * 5.0), 0.0, 1.0)
    grain_penalty = np.clip((grain - ref_grain) / (ref_grain * 5.0), 0.0, 1.0)
    
    shadow_penalty = np.clip(shadow, 0.0, 1.0)
    highlight_penalty = np.clip(highlight, 0.0, 1.0)
    
    snr_penalty = np.clip((ref_snr - snr) / ref_snr, 0.0, 1.0)
    lumvar_penalty = np.clip((ref_lumvar - lumvar) / ref_lumvar, 0.0, 1.0)

    weights = {
        "blur": 0.30,
        "highlight": 0.25,
        "shadow": 0.15,
        "lumvar": 0.15,
        "snr": 0.10,
        "grain": 0.05
    }
    
    weighted_penalty = (
        (blur_penalty * weights["blur"]) +
        (highlight_penalty * weights["highlight"]) +
        (shadow_penalty * weights["shadow"]) +
        (lumvar_penalty * weights["lumvar"]) +
        (snr_penalty * weights["snr"]) +
        (grain_penalty * weights["grain"])
    )
    fairness_score = 1.0 - weighted_penalty
    
    raw_score = float(np.clip(fairness_score, 0.0, 1.0))
    
    # return get_user_friendly_score(raw_score)
    return raw_score * 100

def get_user_friendly_score(raw_score):
    if raw_score >= 0.90:
        return float(90 + (raw_score - 0.90) * 100)
    elif raw_score >= 0.80:
        return float(60 + (raw_score - 0.80) * 300)
    else:
        return max(0, float((raw_score / 0.80) * 60))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test = []
for img_name in os.listdir("train/images"):
    
    if img_name.startswith('.'):
        continue
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    
    pth = os.path.join("train/images", img_name)
    img = Image.open(pth).convert('RGB')
    test.append(transform(img))

test_vecs = [to_relvec(test_img) for test_img in test]
ref = np.mean(test_vecs, axis=0)
# print(score(to_fairvec(test[0]), ref))
