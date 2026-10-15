# 0 = healthy, 1 = HLB
import torch
import torchvision
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torchvision import transforms, datasets
from torch import nn
import numpy as np
import os
import PIL
from PIL import Image, ImageDraw, ImageFont
from torch.optim.lr_scheduler import CosineAnnealingLR
from reliability import score, ref, to_relvec
from pathlib import Path
from tqdm import tqdm

# data augmentation
train_transforms = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(p=0.5),
    # transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), value='random'), # erase patches of image to mimic leaf damage/blocking shadows
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet dataset constants for normalization
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

scale_info = {
    "small": 28,
    "medium": 14,
    "large": 7
}

class CitrusDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, lbl_dir, img_size=(224,224), transform=None):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.img_size = img_size
        self.transform = transform
        self.img_files = sorted([f for f in os.listdir(img_dir) if not f.startswith('.')])

    def __len__(self):
        return len(self.img_files)

    def create_targets(self, grid_size):
        cells = grid_size * grid_size

        boxes = torch.zeros((cells, 4), dtype=torch.float32)
        cls = torch.zeros((cells,), dtype=torch.float32)
        pos_mask = torch.zeros((cells,), dtype=torch.bool)
        
        return boxes, cls, pos_mask

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        targets = {}
        for scale, grid in scale_info.items():
            box, cls, pos_mask = self.create_targets(grid)
            targets[scale] = {
                "box": box,
                "cls": cls,
                "pos_mask": pos_mask
            }

        label_path = os.path.join(
            self.lbl_dir,
            os.path.splitext(img_name)[0] + ".txt"
        )

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    cls_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:])

                    for scale, grid in scale_info.items():
                        gx = min(int(x * grid), grid - 1)
                        gy = min(int(y * grid), grid - 1)

                        cell = gy * grid + gx

                        targets[scale]["box"][cell] = torch.tensor([x, y, w, h])
                        targets[scale]["cls"][cell] = cls_id
                        targets[scale]["pos_mask"][cell] = True
                        
        return img, {scale: (targets[scale]["box"], targets[scale]["cls"], targets[scale]["pos_mask"]) for scale in scale_info}
    
train_set = CitrusDataset("train_set/train/images", "train_set/train/labels", img_size=(224, 224), transform=train_transforms)
val_set = CitrusDataset("train_set/val/images", "train_set/val/labels", img_size=(224, 224), transform=val_test_transforms)

realtest = []
for img_name in os.listdir("seed_data"):
    
    if img_name.startswith('.'):
        continue
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    
    realpth = os.path.join("seed_data", img_name)
    realimg = Image.open(realpth).convert('RGB')
    realtest.append(val_test_transforms(realimg).unsqueeze(0))
       
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

device = torch.device("cpu")

# model implementation

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU()
    
    def forward(self, x):
        return self.activation(self.batchnorm(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, channels, shortcut=True):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, kernel_size=3, padding=1)
        self.conv2 = ConvBlock(channels, channels, kernel_size=3, padding=1)
        self.add_shortcut = shortcut
    
    def forward(self, x):
        if self.add_shortcut:
            return x + self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x))

class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, num_bottlenecks=1, shortcut=True):
        super().__init__()
        self.hidden_channels = out_channels // 2
        self.conv1 = ConvBlock(in_channels, 2*self.hidden_channels, kernel_size=1)
        self.conv2 = ConvBlock((2 + num_bottlenecks) * self.hidden_channels, out_channels, kernel_size=1)
        self.m = nn.ModuleList(Bottleneck(self.hidden_channels, shortcut=shortcut) for _ in range(num_bottlenecks))
        
    def forward(self, x):
        x_mapped = self.conv1(x)
        branch_a, branch_b = x_mapped.chunk(2, dim=1)
        outputs = [branch_a, branch_b]
        for bottleneck_layer in self.m:
            branch_b = bottleneck_layer(branch_b)
            outputs.append(branch_b)
        out_concat = torch.cat(outputs, dim=1)
        return self.conv2(out_concat)

# spatial pyramid pooling - fast
class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = ConvBlock(in_channels, hidden_channels, 1, 1)
        self.conv2 = ConvBlock(hidden_channels * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    
    def forward(self, x):
        x = self.conv1(x)
        
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        
        out = torch.cat((x, y1, y2, y3), dim=1)
        return self.conv2(out)
    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False), nn.ReLU(), nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        
        max_val = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        max_out = self.fc(max_val)
        out = avg_out + max_out
        return self.sigmoid(out)
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.channel_atten = ChannelAttention(in_planes, ratio)
        self.spatial_atten = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = x * self.channel_atten(x)
        x = x * self.spatial_atten(x)
        return x

class DetectHead(nn.Module):
    def __init__(self, in_channels, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.cbam = CBAM(in_planes=in_channels, ratio=16)
        
        self.asym_bottleneck = nn.Sequential(
            nn.Conv2d(in_channels*3, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p=0.2)
        )
        
        self.cls_branch = nn.Sequential(
            ConvBlock(128, 128, kernel_size=3, padding=1),
            ConvBlock(128, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, self.num_classes, kernel_size=1)
        )
        
        self.reg_branch = nn.Sequential(
            ConvBlock(128, 128, kernel_size=3, padding=1),
            ConvBlock(128, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 4, kernel_size=1)
        )
        
    def forward(self, x):
        # asymmetric feature extractor
        x_atten = self.cbam(x)
        asym_h = torch.abs(x_atten - torch.flip(x_atten, dims=[-1]))
        asym_v = torch.abs(x_atten - torch.flip(x_atten, dims=[-2]))
        
        fused = torch.cat([x_atten, asym_h, asym_v], dim=1)
        fused = self.asym_bottleneck(fused)
        
        boxes = self.reg_branch(fused)
        classes = self.cls_branch(fused)
        
        B = x.size(0)
        boxes = boxes.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)
        classes = classes.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
        
        return boxes, classes
    
class CitraScan(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        # custom YOLO
        
        self.layer0 = ConvBlock(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.layer1 = ConvBlock(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.layer2 = C2f(in_channels=32, out_channels=32, num_bottlenecks=1, shortcut=True)
        self.layer3 = ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        
        self.layer4 = C2f(in_channels=64, out_channels=64, num_bottlenecks=2, shortcut=True)
        
        self.layer5 = ConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        
        self.layer6 = C2f(in_channels=128, out_channels=128, num_bottlenecks=2, shortcut=True)
        
        self.layer7 = ConvBlock(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.layer8 = C2f(in_channels=256, out_channels=256, num_bottlenecks=1, shortcut=True)
        
        self.layer9 = SPPF(in_channels=256, out_channels=256, kernel_size=5)
        
        self.small_head  = DetectHead(in_channels=64,  num_classes=self.num_classes) 
        self.medium_head = DetectHead(in_channels=128, num_classes=self.num_classes)
        self.large_head  = DetectHead(in_channels=256, num_classes=self.num_classes)
        
    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat_small = self.layer4(x)
        x = self.layer5(feat_small)
        feat_medium = self.layer6(x)
        x = self.layer7(feat_medium)
        x = self.layer8(x)
        feat_large = self.layer9(x)
        
        out_small = self.small_head(feat_small)
        out_medium = self.medium_head(feat_medium)
        out_large = self.large_head(feat_large)
                
        return {
                    "small": out_small,
                    "medium": out_medium,
                    "large": out_large
                }

class DetectionLoss(nn.Module):
    def __init__(self, box_weight=3.0, cls_weight=1.5, alpha=0.5, beta=3.0):
        super().__init__()
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.alpha = alpha
        self.beta = beta

        self.cls = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def decode_boxes(self, pred_boxes, grid_size):
        B, N, _ = pred_boxes.shape
        device = pred_boxes.device
        
        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_size, device=device),
            torch.arange(grid_size, device=device),
            indexing="ij"
        )
        grid_x = grid_x.reshape(-1)
        grid_y = grid_y.reshape(-1)

        tx = pred_boxes[..., 0]
        ty = pred_boxes[..., 1]
        tw = pred_boxes[..., 2]
        th = pred_boxes[..., 3]

        stride = 1.0 / grid_size
        cx = (grid_x + torch.sigmoid(tx) * 2.0 - 0.5) * stride
        cy = (grid_y + torch.sigmoid(ty) * 2.0 - 0.5) * stride

        tw = torch.clamp(tw, min=-4.0, max=4.0)
        th = torch.clamp(th, min=-4.0, max=4.0)
        
        w = torch.exp(tw) * stride
        h = torch.exp(th) * stride
        
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def xywh_to_xyxy(self, boxes):
        # [x, y, w, h] -> [x1, y2, x2, y2]
        cx, cy, w, h = boxes.unbind(dim=-1)
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
    
    def forward(self, pred_boxes, pred_classes, target_boxes, target_classes, pos_mask, grid_size, epoch=1):
        device = pred_classes.device
        B, N, C = pred_classes.shape
        
        cls_targets = torch.zeros_like(pred_classes)
        
        if pos_mask.sum() > 0:
            t_cls = target_classes[pos_mask].long()
            cls_targets[pos_mask, t_cls] = 1.0
        
        cls_loss = self.bce(pred_classes, cls_targets).sum() / pos_mask.sum().clamp(min=1.0)
                
        if pos_mask is None or pos_mask.sum() == 0:
            zero_box = torch.tensor(0.0, device=pred_classes.device, requires_grad=True)
            return self.cls_weight * cls_loss, zero_box, cls_loss
        
        pred_boxes = self.decode_boxes(pred_boxes, grid_size)
        target_boxes = self.xywh_to_xyxy(target_boxes)
        
        p_boxes = pred_boxes[pos_mask]
        t_boxes = target_boxes[pos_mask]
        
        p_cls = pred_classes[pos_mask]
        t_cls = target_classes[pos_mask].long()
        
        ciou_loss = torchvision.ops.complete_box_iou_loss(p_boxes, t_boxes, reduction="none")
        
        if epoch <= 5:
            box_loss = ciou_loss.mean()
        else:
            iou = torchvision.ops.box_iou(p_boxes, t_boxes).diagonal()
            iou = torch.clamp(iou, min=1e-6)
            
            cls_prob  = torch.softmax(p_cls, dim=-1)
            t_cls = torch.clamp(t_cls, min=0, max=C - 1)
            target_cls_prob = cls_prob.gather(dim=1, index=t_cls.unsqueeze(-1)).squeeze(-1) 
                
            align = (target_cls_prob ** self.alpha) * (iou ** self.beta)
            align = align.detach()
            box_loss = (align * ciou_loss).sum() / (align.sum() + 1e-6)

        total_loss = (self.box_weight * box_loss) + (self.cls_weight * cls_loss)
        
        return total_loss, box_loss, cls_loss

# class DetectionLoss(nn.Module):
#     def __init__(self, box_weight=3.0, cls_weight=1.5, alpha=0.5, beta=3.0):
#         super().__init__()
#         self.box_weight = box_weight
#         self.cls_weight = cls_weight
#         self.alpha = alpha
#         self.beta = beta

#         self.ce = nn.CrossEntropyLoss(reduction="none")

#     def decode_boxes(self, pred_boxes, grid_size):
#         B, N, _ = pred_boxes.shape
#         device = pred_boxes.device
        
#         grid_y, grid_x = torch.meshgrid(
#             torch.arange(grid_size, device=device),
#             torch.arange(grid_size, device=device),
#             indexing="ij"
#         )
#         grid_x = grid_x.reshape(-1)
#         grid_y = grid_y.reshape(-1)

#         tx, ty, tw, th = pred_boxes[..., 0], pred_boxes[..., 1], pred_boxes[..., 2], pred_boxes[..., 3]
#         stride = 1.0 / grid_size
        
#         cx = (grid_x + torch.sigmoid(tx) * 2.0 - 0.5) * stride
#         cy = (grid_y + torch.sigmoid(ty) * 2.0 - 0.5) * stride
#         tw = torch.clamp(tw, min=-4.0, max=4.0)
#         th = torch.clamp(th, min=-4.0, max=4.0)
        
#         w = torch.exp(tw) * stride
#         h = torch.exp(th) * stride
        
#         x1, y1 = cx - w / 2.0, cy - h / 2.0
#         x2, y2 = cx + w / 2.0, cy + h / 2.0
        
#         return torch.stack([x1, y1, x2, y2], dim=-1)

#     def xywh_to_xyxy(self, boxes):
#         cx, cy, w, h = boxes.unbind(dim=-1)
#         return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
    
#     def forward(self, pred_boxes, pred_classes, target_boxes, target_classes, pos_mask, grid_size, epoch=1):
#         device = pred_classes.device
#         B, N, C = pred_classes.shape
        
#         cls_targets = torch.zeros((B, N), dtype=torch.long, device=device)
        
#         if pos_mask.sum() > 0:
#             t_cls = target_classes[pos_mask].long() + 1
#             cls_targets[pos_mask] = t_cls
        
#         pred_permuted = pred_classes.permute(0, 2, 1)
#         cls_loss_map = self.ce(pred_permuted, cls_targets) # [B, N]
        
#         cls_loss = cls_loss_map.mean()
                
#         if pos_mask is None or pos_mask.sum() == 0:
#             zero_box = torch.tensor(0.0, device=device, requires_grad=True)
#             return self.cls_weight * cls_loss, zero_box, cls_loss
        
#         pred_boxes = self.decode_boxes(pred_boxes, grid_size)
#         target_boxes = self.xywh_to_xyxy(target_boxes)
        
#         p_boxes = pred_boxes[pos_mask]
#         t_boxes = target_boxes[pos_mask]
#         p_cls = pred_classes[pos_mask]
#         t_cls = target_classes[pos_mask].long() + 1
        
#         ciou_loss = torchvision.ops.complete_box_iou_loss(p_boxes, t_boxes, reduction="none")
        
#         if epoch <= 5:
#             box_loss = ciou_loss.mean()
#         else:
#             iou = torchvision.ops.box_iou(p_boxes, t_boxes).diagonal()
#             iou = torch.clamp(iou, min=1e-6)
            
#             cls_prob = torch.softmax(p_cls, dim=-1)
#             target_cls_prob = cls_prob.gather(dim=1, index=t_cls.unsqueeze(-1)).squeeze(-1) 
                
#             align = (target_cls_prob ** self.alpha) * (iou ** self.beta)
#             align = align.detach()
#             box_loss = (align * ciou_loss).sum() / (align.sum() + 1e-6)

#         total_loss = (self.box_weight * box_loss) + (self.cls_weight * cls_loss)
        
#         return total_loss, box_loss, cls_loss

class MultiScaleDetectionLoss(nn.Module):
    def __init__(self, box_weight=3.0, cls_weight=1.5):
        super().__init__()

        self.single_loss = DetectionLoss(box_weight=box_weight, cls_weight=cls_weight)

    def forward(self, preds_dict, targets_dict):

        total_loss = 0.0
        box_loss_sum = 0.0
        cls_loss_sum = 0.0

        for scale in ["small", "medium", "large"]:
            grid_size = scale_info[scale]
            p_box, p_cls = preds_dict[scale]
            t_box, t_cls, pos_mask = targets_dict[scale]

            loss, box_loss, cls_loss = self.single_loss(p_box, p_cls, t_box, t_cls, pos_mask, grid_size)

            total_loss += loss
            box_loss_sum += box_loss
            cls_loss_sum += cls_loss

        return total_loss, box_loss_sum, cls_loss_sum

@torch.no_grad()
def infer(model, img, conf=0.60, iou=0.45, device="cpu"):
    model.eval()
    ow, oh = img.size

    input_tensor = val_test_transforms(img.copy()).unsqueeze(0).to(device)
    preds_dict = model(input_tensor)

    all_boxes_px, all_scores, all_cls_ids = [], [], []

    for scale_name, grid_size in scale_info.items():
        pred_boxes, pred_classes = preds_dict[scale_name]
        
        box_logits = pred_boxes.squeeze(0)
        cls_logits = pred_classes.squeeze(0)

        cls_probs = torch.sigmoid(cls_logits) 
        scores, cls_ids = torch.max(cls_probs, dim=-1)

        valid_mask = scores >= conf
        if valid_mask.sum() == 0:
            continue

        box_logits = box_logits[valid_mask]
        scores = scores[valid_mask]
        cls_ids = cls_ids[valid_mask]

        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_size, device=device),
            torch.arange(grid_size, device=device),
            indexing="ij"
        )
        grid_x = grid_x.reshape(-1)[valid_mask]
        grid_y = grid_y.reshape(-1)[valid_mask]

        tx, ty, tw, th = box_logits[:, 0], box_logits[:, 1], box_logits[:, 2], box_logits[:, 3]

        stride = 1.0 / grid_size
        cx = (grid_x.float() + torch.sigmoid(tx) * 2.0 - 0.5) * stride
        cy = (grid_y.float() + torch.sigmoid(ty) * 2.0 - 0.5) * stride

        w = torch.exp(torch.clamp(tw, min=-4.0, max=4.0)) * stride
        h = torch.exp(torch.clamp(th, min=-4.0, max=4.0)) * stride

        boxes = torch.zeros_like(box_logits)
        
        x0 = (cx - w / 2) * ow
        y0 = (cy - h / 2) * oh
        x1 = (cx + w / 2) * ow
        y1 = (cy + h / 2) * oh

        boxes[:, 0] = torch.min(x0, x1)
        boxes[:, 1] = torch.min(y0, y1)
        boxes[:, 2] = torch.max(x0, x1)
        boxes[:, 3] = torch.max(y0, y1)

        boxes[:, 0].clamp_(min=0, max=ow)
        boxes[:, 1].clamp_(min=0, max=oh)
        boxes[:, 2].clamp_(min=0, max=ow)
        boxes[:, 3].clamp_(min=0, max=oh)

        valid_area_mask = (boxes[:, 2] > boxes[:, 0] + 2) & (boxes[:, 3] > boxes[:, 1] + 2)
        if valid_area_mask.sum() == 0:
            continue

        all_boxes_px.append(boxes[valid_area_mask])
        all_scores.append(scores[valid_area_mask])
        all_cls_ids.append(cls_ids[valid_area_mask])

    if len(all_boxes_px) == 0:
        return None, None, None

    cat_boxes = torch.cat(all_boxes_px, dim=0)
    cat_scores = torch.cat(all_scores, dim=0)
    cat_cls_ids = torch.cat(all_cls_ids, dim=0)

    nms_idcs = torchvision.ops.nms(cat_boxes, cat_scores, iou)

    return cat_boxes[nms_idcs], cat_scores[nms_idcs], cat_cls_ids[nms_idcs]



def predict(model, img_path, out_path, conf=0.50, iou=0.45, device="cpu"):
    mapping = {0: "Healthy", 1: "HLB"}

    oimg = Image.open(img_path)
    oimg = PIL.ImageOps.exif_transpose(oimg).convert("RGB")
    ow, oh = oimg.size

    boxes, scores, cls_ids = infer(model, oimg, conf=conf, iou=iou, device=device)

    if boxes is None or len(boxes) == 0:
        print(f"No detections above confidence threshold ({conf})")
        oimg.save(out_path)
        return

    draw = ImageDraw.Draw(oimg)
    font_size = max(16, int(oh * 0.025))
    large_font = ImageFont.load_default(size=font_size)
    
    for box, score, cls_id in zip(boxes, scores, cls_ids):

        x1, y1, x2, y2 = box.tolist()

        cls_name = mapping[int(cls_id.item())]
        confidence = score.item() * 100

        print(f"detected: {cls_name} | confidence: {confidence:.2f}%")

        box_color = (255, 125, 125) if cls_id.item() == 1 else (0, 250, 255)
        text_color = (0, 0, 0)

        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=5)
        
        label_text = f"{cls_name} {confidence:.1f}%"
        left, top, right, bottom = large_font.getbbox(label_text)
        text_w = right - left
        text_h = bottom - top
        padding = 8
        
        if y1 - (text_h + padding * 2) >= 0:
            bg_y1 = y1 - (text_h + padding * 2)
            bg_y2 = y1
        else:
            bg_y1 = y1
            bg_y2 = y1 + text_h + padding * 2
        
        bg_x1 = x1
        bg_x2 = x1 + text_w + padding * 2
        
        draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=box_color)

        draw.text((bg_x1 + padding, bg_y1 + padding - top), label_text, fill=text_color, font=large_font)
        
    oimg.save(out_path)

def evaluate_test_accuracy(model, test_loader, conf=0.50, iou=0.45, device="cpu"):
    model.eval()

    correct_class = 0
    total_images = 0
    detections_found = 0

    confusion_matrix = np.zeros((2, 2), dtype=int)

    print("Evaluating Test Dataset...")

    for images, targets in tqdm(test_loader):
        batch_size = images.size(0)

        for i in range(batch_size):
            # find ground truth cls from multi scale targets
            gt_class = 0
            for scale_name, (box_target, cls_target, pos_mask) in targets.items():
                pos_indices = torch.nonzero(pos_mask[i])
                if len(pos_indices) > 0:
                    first_pos = pos_indices[0]
                    if first_pos.numel() == 2:
                        grid_y, grid_x = first_pos[0].item(), first_pos[1].item()
                        gt_cls = cls_target[i, grid_y, grid_x]
                    else:
                        flat_idx = first_pos.item()
                        gt_cls = cls_target[i, flat_idx]

                    gt_class = int(gt_cls.item()) if (gt_cls.ndim == 0 or gt_cls.numel() == 1) else int(torch.argmax(gt_cls).item())
                    break

            img_tensor = images[i].cpu()
            img_np = img_tensor.permute(1, 2, 0).numpy()
            img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            boxes, scores, cls_ids = infer(model, pil_img, conf=conf, iou=iou, device=device)

            total_images += 1

            if boxes is not None and len(boxes) > 0:
                detections_found += 1
                best_idx = torch.argmax(scores)
                pred_class = int(cls_ids[best_idx].item())

                if pred_class == gt_class:
                    correct_class += 1

                confusion_matrix[gt_class][pred_class] += 1
            else:
                print(f"Warning: Image {total_images} had no detections above conf threshold {conf}")

    # final accuracy metrics
    cls_accuracy = (correct_class / detections_found * 100) if detections_found > 0 else 0
    detection_rate = (detections_found / total_images * 100)

    print("\n" + "="*45)
    print("      TEST DATASET EVALUATION SUMMARY      ")
    print("="*45)
    print(f"Total Test Images    : {total_images}")
    print(f"Detections Found     : {detections_found} ({detection_rate:.1f}%)")
    print(f"Classification Acc   : {cls_accuracy:.2f}%")
    print("-" * 45)
    print("Confusion Matrix:")
    print("                 Pred Healthy  Pred HLB")
    print(f"True Healthy :      {confusion_matrix[0][0]:<10} {confusion_matrix[0][1]:<10}")
    print(f"True HLB     :      {confusion_matrix[1][0]:<10} {confusion_matrix[1][1]:<10}")
    print("="*45)

    return cls_accuracy
    
    
model = CitraScan().to(device)

if os.path.isfile("fine_tuned_anchor-free2.pt"):
    model = CitraScan(num_classes=3)
    checkpoint = torch.load("fine_tuned_anchor-free2.pt", map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    # cal_pth = Path("other_data/lemon_healthy")
    # valid_extensions = {".jpg", ".JPG"}
    # cal_list = []
    
    # for pth in cal_pth.iterdir():
    #     if pth.suffix.lower() in valid_extensions:
    #         with Image.open(pth) as img:
    #             cal_list.append(img.copy())
                
    # cal_set = torch.stack([val_test_transforms(img) for img in cal_list])
    # cal_lbls = torch.zeros(cal_set.shape[0])
    
    # temp = nn.Parameter(torch.ones(1))
    # optimizer = torch.optim.LBFGS([temp], lr=0.01)
    
    # outs = model(cal_set)
    # print(outs)
    
    # def eval():
    #     optimizer.zero_grad()
        # loss = nn.CrossEntropyLoss()(logits / temp)
    # model.load_state_dict(torch.load("weights/best_anchor-free1.pt", map_location=device))

    predict(model=model, img_path="IMG_7056.jpg", out_path="img_predres.jpg", conf=0.3, device=device)

    # accuracy = evaluate_test_accuracy(model=model, test_loader=test_loader, conf=0, iou=0.45, device=device)
    # print(f"test acc: {accuracy:.2f}%")

else:
    criterion = MultiScaleDetectionLoss().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    output_dir = Path("weights")
    output_dir.mkdir(exist_ok=True)

    best_val = float("inf")
    epochs=100

    for epoch in range(1, epochs + 1):
        model.train()
        rtotal_loss = 0.0
        rbox_loss = 0.0
        rcls_loss = 0.0
        
        for batch_idx, (features, targets_dict) in enumerate(train_loader):
            features = features.to(device)
            
            targets_dict = {
                scale: tuple(t.to(device) for t in scale_targets) for scale, scale_targets in targets_dict.items()
            }
            
            optimizer.zero_grad()
            
            preds_dict = model(features)
            loss, box_loss, cls_loss = criterion(preds_dict, targets_dict)
            
            loss.backward()
            optimizer.step()
            
            rtotal_loss += loss.item()
            rbox_loss += box_loss.item()
            rcls_loss += cls_loss.item()
        
        scheduler.step()
        
        aloss = rtotal_loss / len(train_loader)
        abox_loss = rbox_loss / len(train_loader)
        acls_loss = rcls_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets_dict in val_loader:
                features = features.to(device)
                targets_dict = {
                    scale: tuple(t.to(device) for t in scale_targets) for scale, scale_targets in targets_dict.items()
                }
                
                preds_dict = model(features)
                loss, _, _ = criterion(preds_dict, targets_dict)
                val_loss += loss.item()
        
        aval_loss = val_loss / len(val_loader)
        
        print(
            f"Epoch [{epoch:02d}/{epochs}] | "
            f"Train Loss: {aloss:.4f} (Box: {abox_loss:.4f}, Cls: {acls_loss:.4f}) | "
            f"Val Loss: {aval_loss:.4f}"
        )
        
        if aval_loss < best_val:
            best_val= aval_loss
            save_path = output_dir / "best_anchor-free1.pt"
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model to {save_path} with val loss: {best_val:.4f}")

    torch.save(model.state_dict(), output_dir / "last_anchor-free1.pt")