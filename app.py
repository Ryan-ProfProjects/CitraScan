import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
from modelwyolo import val_test_transforms, segmodel, clsmodel

segstate_dict = torch.load('leaf_segmentor2.pth', weights_only=True)
clsstate_dict = torch.load("leaf_classifier.pth", weights_only=True)
segmodel.load_state_dict(segstate_dict)
clsmodel.load_state_dict(clsstate_dict)
segmodel.eval()
clsmodel.eval()

labels = ["Healthy / Nutrient Deficiency", "Suspected HLB (Huanglongbing)"]

def predict_citrus(img):
    if img is None:
        return "Please upload an image."

    features = val_test_transforms(img).unsqueeze(0)
    
    with torch.no_grad():
        backbone, mask, _ = segmodel(features)
        logits = clsmodel(backbone, mask)
        probs = torch.nn.functional.softmax(logits[0], dim=0)
    
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

# web UI layout
description_text = """
### ⚠️ EDUCATIONAL SCREENING TOOL ONLY
This AI model is an experimental tool designed for preliminary screening and educational purposes. 
It does not provide official regulatory diagnoses. For official confirmation or to report suspected 
cases in California, please immediately contact the CDFA Exotic Pest Hotline at 1-800-491-1899.
"""

interface = gr.Interface(
    fn=predict_citrus,
    inputs=gr.Image(type="pil", label="Take or upload a clear photo of the suspicious leaf"),
    outputs=gr.Label(num_top_classes=2, label="Prediction Probability"),
    title="Open-Source Citrus HLB Detector",
    description=description_text,
    flagging_mode="auto", # automatically logs incoming real-world data to local CSV/image folder
    flagging_dir="hlb_real_world_dataset" 
)
interface.launch()