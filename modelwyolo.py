# 0 = healthy, 1 = HLB
import torch
import torchvision
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import os
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from pipeline import score, ref, to_fairvec
    
# data augmentation
train_transforms = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.ColorJitter(brightness=0.6, contrast=0.5, saturation=0.5, hue=0.2),
    transforms.RandomGrayscale(p=0.2),
    # transforms.RandomSolarize(threshold=192, p=0.2),
    # transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    # transforms.GaussianBlur(kernel_size=(3,3), sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), value='random'), # erase patches of image to mimic leaf damage/blocking shadows
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet dataset constants for normalization
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CitrusDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, lbl_dir, img_size=(224, 224), mask_size=(56, 56), transform=None):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.img_size = img_size
        self.transform = transform
        self.mask_size = mask_size
        
        self.img_files = sorted([f for f in os.listdir(img_dir) if not f.startswith('.')])

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        pth = os.path.join(self.img_dir, img_name)
        img = Image.open(pth).convert("RGB")
                
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
            
        img = torchvision.transforms.functional.resize(img, self.img_size)
            
        h_mask, w_mask = self.mask_size
        mask = torch.zeros((h_mask, w_mask), dtype=torch.float32)
        lbl_name = os.path.splitext(img_name)[0] + ".txt"
        lbl_pth = os.path.join(self.lbl_dir, lbl_name)
        
        global_lbl = 1 if "HLB" in self.img_dir.upper() or "HLB" in img_name.upper() else 0
        
        if os.path.exists(lbl_pth):
            with open(lbl_pth, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    classid = int(parts[0])
                    if classid == 1:
                        global_lbl = 1
                    
                    x_cen, y_cen, w, h = map(float, parts[1:])
                    
                    x1 = int((x_cen - w / 2) * w_mask)
                    y1 = int((y_cen - h / 2) * h_mask)
                    x2 = int((x_cen + w / 2) * w_mask)
                    y2 = int((y_cen + h / 2) * h_mask)
                    
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_mask, x2), min(h_mask, y2)

                    mask[y1:y2, x1:x2] = 1.0 # all the coordinates inside mask are activated
        mask = mask.unsqueeze(0)
        return img, global_lbl, mask
        
    def __len__(self):
        return len(self.img_files)

train_set = CitrusDataset("train/images", "train/labels", img_size=(224, 224), mask_size=(56, 56), transform=train_transforms)
val_set = CitrusDataset("valid/images", "valid/labels", img_size=(224, 224), mask_size=(56, 56), transform=val_test_transforms)
test_set = CitrusDataset("test/images", "test/labels", img_size=(224, 224), mask_size=(56, 56), transform=val_test_transforms)

testimg = Image.open("healthy-citrus-leaves-jpg-noelle-johnson-landscape-consulting-img~4b71c74301143308_14-0176-1-e5fed3a.jpg").convert("RGB")
testimg = val_test_transforms(testimg).unsqueeze(0)

realtest = []
for img_name in os.listdir("seed_data"):
    
    if img_name.startswith('.'):
        continue
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    
    realpth = os.path.join("seed_data", img_name)
    realimg = Image.open(realpth).convert('RGB')
    realtest.append(val_test_transforms(realimg).unsqueeze(0))
    
# print(score(to_fairvec(testimg.squeeze(0)), ref))

       
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

device = torch.device("mps")

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
    
class Segmentor(nn.Module):
    def __init__(self, num_masks=32):
        super().__init__()
        
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
        
        self.mask_conv = nn.Conv2d(in_channels=49, out_channels=1, kernel_size=1)
        
        self.proto_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, num_masks, 1)
        )
        
        self.bbox_head = nn.Conv2d(256, 4, 1) 
        self.coeff_head = nn.Conv2d(256, num_masks, 1)
        
    def forward(self, x, detach=False):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        
        prototypes = self.proto_head(x)
        
        # standard detection fields
        pred_boxes = self.bbox_head(x)
        pred_coeffs = self.coeff_head(x)
        
        # construct masks
        batch_size, num_masks, h, w = prototypes.shape
        proto = prototypes.view(batch_size, num_masks, h * w)
        coeffs = pred_coeffs.view(batch_size, num_masks, h * w).permute(0, 2, 1) # permute for matmul
                
        pred_masks = torch.bmm(coeffs, proto) # batch matrix mult
        pred_masks = pred_masks.view(batch_size, 49, 7, 7)
        
        oglobal_masks = self.mask_conv(pred_masks)

        global_masks = torch.nn.functional.interpolate(oglobal_masks, size=(56, 56), mode="bilinear", align_corners=False)
                
        return x, oglobal_masks, global_masks
            
class Classifier(nn.Module):
    def __init__(self, num_masks=32):
        super().__init__()
        self.smoother = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        nn.init.constant_(self.smoother.weight, 1.0 / 9.0)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.clsconv1 = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.clsconv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_features=7*7*32, out_features=2, bias=True)
        
    def forward(self, x, global_mask):
        # print(x.shape, global_mask.shape)
        smooth_mask = self.smoother(global_mask)
        x_masked = x * torch.sigmoid(smooth_mask)
        
        B, C, H, W = x_masked.shape
        x_flat = x_masked.view(B, C, H * W).permute(0, 2, 1)
        x_relations = self.transformer(x_flat)
        x_atten = x_relations.permute(0, 2, 1).view(B, C, H, W)
        
        out = torch.relu(self.clsconv1(x_atten))
        out = torch.relu(self.clsconv2(out))
        
        out = torch.nn.functional.adaptive_avg_pool2d(out, (7, 7))
        
        out = torch.flatten(out, start_dim=1)
        out = self.dropout(out)
        logits = self.fc(out)
        
        return logits
    
    
segmodel = Segmentor()
clsmodel = Classifier()
segmodel.to(device)
clsmodel.to(device)

# if os.path.isfile("leaf_classifier.pth"):
#     segstate_dict = torch.load('leaf_segmentor2.pth', weights_only=True)
#     clsstate_dict = torch.load("leaf_classifier.pth", weights_only=True)
#     segmodel.load_state_dict(segstate_dict)
#     clsmodel.load_state_dict(clsstate_dict)
#     segmodel.eval()
#     clsmodel.eval()
    
#     test_loader = DataLoader(test_set, batch_size=32, shuffle=True)
#     corr = 0
#     hlbcounts = 0
#     for batchidx, (features, labels, target_masks) in enumerate(test_loader):        
#         features = features.to(device)
#         labels = labels.to(device)
#         backbone, mask, _ = segmodel(features)
#         logits = clsmodel(backbone, mask)
#         probs = torch.softmax(logits, dim=1).squeeze(0)
#         preds = torch.argmax(probs, dim=1)
#         for pred in preds:
#             if pred == 1:
#                 hlbcounts+=1
                
#         correct_preds = torch.sum((labels==preds).int())
#         print(f"batch {batchidx} acc: {correct_preds.item()/len(labels)}")
#         corr += correct_preds.item()
#     print(f"Test acc: {corr/len(test_set):.4f}")
#     print(f"HLB percentage: {hlbcounts/len(test_set):4f}")
    
#     argmap = {0: "healthy", 1: "HLB"}
#     backbone, mask, _ = segmodel(testimg.to(device))
#     logit = clsmodel(backbone, mask)
#     probs = torch.softmax(logit, dim=1)
#     # print(probs)
#     confidence, pred = torch.max(probs, 1)
#     confidence = confidence.item() * 100
#     disease_status = argmap[pred.item()]
#     print(f"Disease status: {disease_status}")
#     if disease_status == "healthy":
#         print(f"{(probs[0][0]*100):.2f}% total healthy confidence")
#     if disease_status == "HLB":
#         print(f"{(probs[0][1]*100):.2f}% total HLB confidence")
#     print(f"{confidence:.2f}% overall diagnostic confidence")
    
#     hcounts = 0
#     hlbcounts = 0
#     for testimg in realtest:
#         backbone, mask, _ = segmodel(testimg.to(device))
#         logit = clsmodel(backbone, mask)
#         prob = torch.softmax(logit, dim=1).squeeze(0)
#         pred = torch.argmax(logit, dim=1)
#         print(pred)
#         if pred == 0:
#             hcounts+=1
#         if pred == 1:
#             hlbcounts+=1
#     print(hlbcounts)
#     print(f"Real acc: {hcounts/len(realtest):.4f}")


# # for training
# if os.path.isfile("leaf_segmentor2.pth"):
#     # classifier

#     epochs = 100
    
#     celoss = torch.nn.CrossEntropyLoss()

#     clsoptimizer = torch.optim.AdamW(clsmodel.parameters(), lr=3e-4, weight_decay=0.01)
#     total_steps = epochs * len(train_loader)
#     scheduler = CosineAnnealingLR(clsoptimizer, T_max=total_steps, eta_min=1e-6)

#     loss_history = []
#     val_history = []

#     segmodel.load_state_dict(torch.load('leaf_segmentor2.pth', map_location=device))
#     segmodel.eval() 
#     for param in segmodel.parameters():
#         param.requires_grad = False
        
#     best_val = float('inf')
#     for epoch in range(epochs):
#         clsmodel.train()
#         running_loss = 0
#         for batchidx, (features, labels, _) in enumerate(train_loader):
#             features = features.to(device)
#             labels = labels.to(device)
            
#             clsoptimizer.zero_grad()
            
#             backbone, oglobal_mask, _ = segmodel(features)
            
#             logits = clsmodel(backbone, oglobal_mask)

#             # loss = 1.0 * clsloss + 1.0 * mloss + 0.1 * mask_size_loss + 0.1 * variance_loss + 0.2 * tv_loss
#             loss = celoss(logits, labels)
#             print(f"CLS: {loss.item():.4f}")
            
#             loss.backward()
#             clsoptimizer.step()  
#             scheduler.step()
            
#             running_loss += loss.item()
    
#         loss_history.append(running_loss / len(train_loader)) 
#         clsmodel.eval()
#         running_val = 0
#         with torch.no_grad():
#             for val_feats, val_lbls, _ in val_loader:
#                 val_feats = val_feats.to(device)
#                 val_lbls = val_lbls.to(device)
                
#                 val_backbone, val_masks, _ = segmodel(val_feats)
#                 val_logits = clsmodel(val_backbone, val_masks)
#                 vloss = celoss(val_logits, val_lbls)
#                 running_val += vloss.item()
#         avg_val = running_val / len(val_loader)
#         val_history.append(avg_val)
        
#         if avg_val < best_val:
#             best_val = avg_val
#             torch.save(clsmodel.state_dict(), 'best_classifier.pth')
#             print(f"saved new best model at epoch {epoch} with val loss: {avg_val:.4f}")
        
#         if epoch % 10 == 0:
#             avgloss = running_loss / len(train_loader)
#             print(f"epoch {epoch} avg loss {avgloss:.4f}")

#     plt.plot(range(epochs), loss_history, label='train', color='blue')
#     plt.plot(range(epochs), val_history, label='val', color='orange')
#     plt.xlabel('epoch')
#     plt.ylabel('loss')
#     plt.legend()
#     plt.savefig('loss_curve_classifier.png')
#     torch.save(clsmodel.state_dict(), 'leaf_classifier.pth')

# else:    
#     epochs = 100

#     maskloss = nn.BCEWithLogitsLoss()
#     segoptimizer = torch.optim.AdamW(segmodel.parameters(), lr=3e-4, weight_decay=0.01)
    
#     total_steps = epochs * len(train_loader)
#     scheduler = CosineAnnealingLR(segoptimizer, T_max=total_steps, eta_min=1e-6)

#     loss_history = []
#     val_history = []
    
#     for epoch in range(epochs):
#         segmodel.train()
#         running_loss = 0
#         for batchidx, (features, _, target_masks) in enumerate(train_loader):
#             features = features.to(device)
            
#             segoptimizer.zero_grad()
            
#             _, _, global_masks = segmodel(features)
                        
#             # # construct masks
#             # batch_size, num_masks, h, w = prototypes.shape
#             # num_boxes = pred_coeffs.shape[1]
#             # proto = prototypes.view(batch_size, num_masks, h * w)
#             # coeffs = pred_coeffs.view(batch_size, num_masks, h * w).permute(0, 2, 1)
#             # masks = torch.bmm(coeffs, proto) # batch matrix mult
#             # pred_masks = masks.view(batch_size, -1, h, w) # reshape back
#             # pred_masks = segmodel.mask_conv(pred_masks)
#             # pred_masks = torch.nn.functional.interpolate(pred_masks, size=(56, 56), mode="bilinear", align_corners=False)
            
#             global_masks = global_masks.to(device)
#             target_masks = target_masks.to(device)
            
#             mloss = maskloss(global_masks, target_masks)
            
#             # target_coverage = 0.25
#             # current_coverage = global_masks.mean()
#             # mask_size_loss = torch.abs(target_coverage - current_coverage)
#             # variance_loss = torch.mean(global_masks * (1-global_masks))
#             # # total variance loss to smooth out local variance:
#             # diff_i = torch.abs(global_masks[:, :, :, 1:] - global_masks[:, :, :, :-1]) # horizontal shift 1 pixel diff
#             # diff_j = torch.abs(global_masks[:, :, 1:, :] - global_masks[:, :, :-1, :]) # vertical
#             # tv_loss = diff_i.mean() + diff_j.mean()
#             # loss = 1.0 * clsloss + 1.0 * mloss + 0.1 * mask_size_loss + 0.1 * variance_loss + 0.2 * tv_loss
            
#             loss = mloss
#             print(f"Mask: {mloss.item():.4f}")
            
#             loss.backward()
#             segoptimizer.step()
#             scheduler.step()  
            
#             running_loss += loss.item()
    
#         loss_history.append(running_loss / len(train_loader))
        
#         segmodel.eval()
#         running_val = 0
#         with torch.no_grad():
#             for val_feats, _, val_targets in val_loader:
#                 val_feats = val_feats.to(device)
#                 val_targets = val_targets.to(device)
                
#                 _, _, val_masks = segmodel(val_feats)
#                 vloss = maskloss(val_masks, val_targets)
#                 running_val += vloss.item()
#         val_history.append(running_val / len(val_loader))
                
#         if epoch % 10 == 0:
#             avgloss = running_loss / len(train_loader)
#             print(f"epoch {epoch} avg loss {avgloss:.4f}")

#     plt.plot(range(epochs), loss_history, label='train', color='blue')
#     plt.plot(range(epochs), val_history, label='val', color='orange')
#     plt.xlabel('epoch')
#     plt.ylabel('loss')
#     plt.legend()
#     plt.savefig('loss_curve_segmentor.png')
#     torch.save(segmodel.state_dict(), 'leaf_segmentor2.pth')
