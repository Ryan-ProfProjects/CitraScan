import gradio as gr
import torch
from PIL import Image, ImageOps, ImageDraw, ImageFont

# import infer from anchor_free
from anchor_free import infer, CitraScan, val_test_transforms
from reliability import ref, to_relvec, score

model = CitraScan(num_classes=3)
# for head in [model.small_head, model.medium_head, model.large_head]:
#     head.num_classes = 3
    # head.cls_branch[-1] = torch.nn.Conv2d(128, 3, kernel_size=1)
    
mapping = {0: "Healthy", 1: "HLB", 2: "Nutrient Deficiency"}

def draw_yolo_boxes(img_pil, boxes, scores, cls_ids):
    """Draws YOLOv8-style bounding boxes with background labels on PIL image."""
    annotated_img = img_pil.copy()
    ow, oh = annotated_img.size
    draw = ImageDraw.Draw(annotated_img)

    font_size = max(16, int(oh * 0.025))
    font = ImageFont.load_default(size=font_size)

    for box, score, cls_id in zip(boxes, scores, cls_ids):
        x1, y1, x2, y2 = box.tolist()
        cls_name = mapping[int(cls_id.item())]
        confidence = score.item() * 100

        box_color = (255, 125, 125) if cls_id.item() == 1 else (0, 250, 255)
        text_color = (0, 0, 0)

        line_w = max(3, int(oh * 0.005))
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=line_w)

        # label_text = f"{cls_name} {confidence:.1f}%"
        label_text = f"{cls_name}"
        left, top, right, bottom = font.getbbox(label_text)
        text_w, text_h = right - left, bottom - top
        padding = 6

        if y1 - (text_h + padding * 2) >= 0:
            bg_y1, bg_y2 = y1 - (text_h + padding * 2), y1
        else:
            bg_y1, bg_y2 = y1, y1 + text_h + padding * 2

        bg_x1, bg_x2 = x1, x1 + text_w + padding * 2

        draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=box_color)
        draw.text((bg_x1 + padding, bg_y1 + padding - top), label_text, fill=text_color, font=font)

    return annotated_img

def analyze_citrus_leaf(img):
    if img is None:
        return None, "Please upload an image.", "N/A", "N/A"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load("fine_tuned_anchor-free2.pt", map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    img = ImageOps.exif_transpose(img).convert("RGB")

    boxes, scores, cls_ids = infer(model, img, conf=0.1, iou=0.25, device=device)

    if boxes is None or len(boxes) == 0:
        rel_str = "N/A"
        try:
            rel = score(to_relvec(val_test_transforms(img)), ref)
            rel_str = f"{rel:.2f}%"
        except Exception:
            pass
        return img, "No Leaf Detected. Try another picture.", "0.00%", rel_str, gr.update(value="Tip: Ensure there is less background noise and the leaf is centered in the frame.", visible=True)

    annotated_img = draw_yolo_boxes(img, boxes, scores, cls_ids)

    top_idx = torch.argmax(scores)
    disease_status = mapping[int(cls_ids[top_idx].item())]
    confidence = scores[top_idx].item() * 100
    
    try:
        rel = score(to_relvec(val_test_transforms(img)), ref)
        rel_str = f"{rel:.2f}%"
    except Exception:
        rel_str = "N/A"
        
    print(confidence, disease_status, rel)
    
    if confidence > 50.0 and disease_status=="HLB" or disease_status=="Nutrient Deficiency":
        if rel > 75.0:
            rec_output = """
        **Likely Concern detected**
        
        The image contains patterns that may be associated with citrus health issues. Consider consulting citrus health experts.
        """
        else:
            rec_output = """
        **Likely Concern Probable**
        
        The image contains patterns that may be associated with citrus health issues, but can be influenced by poor lighting or conditions the model is not used to. 
        Consider taking another photo in relatively uniform lighting conditions or consult citrus experts.
        """
    elif confidence < 50.0 and disease_status=="HLB" or disease_status=="Nutrient Deficiency":
        if rel > 75.0:
            rec_output = """
        **Possible concern detected**
        
        The image contains patterns that may be associated with citrus health issues. Consider taking another photo or consulting citrus health experts.
        """
        else:
            rec_output = """
        **Probable concerns**
        
        The image contains patterns that may be associated with citrus health issues, but can be influenced by poor lighting the model is not used to. 
        Consider taking another photo in relatively uniform lighting conditions.
        """
    elif confidence > 50.0 and disease_status == "Healthy" and rel > 75.0:
        rec_output = """
    **No concerns detected. Likely healthy**
    
    To be sure, consider taking another photo or consulting citrus health experts.
    """
    else:
        rec_output = """
    **No concerns detected. Potentially healthy**
    
    To be sure, consider taking another photo or consulting citrus health experts.
    """
    
    confidence = f"{scores[top_idx].item() * 100:.2f}%"
    
    note = """
    **About Model Confidence**

    - Confidence is how strongly the model supports the predicted class. It is **not** the probability that the diagnosis is correct.
    - A higher confidence does **not** guarantee the prediction is correct.
    - Poor lighting, blur, unusual viewing angles, or symptoms not represented in the training data can affect reliability. The **Image Reliability** score estimates how similar the uploaded image is to the images the model was trained on, helping indicate whether the model is operating under familiar conditions.
    - This tool is intended for educational screening only and should not replace professional diagnosis.
    """

    return annotated_img, disease_status, rec_output, confidence, rel_str, gr.update(value=note, visible=True)


description_text = "Analyze citrus leaf photos in seconds with Computer Vision for HLB screening"

with gr.Blocks(title="Citrus Greening (HLB) Screening Tool") as demo:
    gr.Markdown("# 🍊 Open-Source Citrus Greening (HLB) Screening Tool")
    gr.Markdown(description_text)

    with gr.Row():
        with gr.Column():
            uploaded_file = gr.Image(type="pil", label="Upload a Citrus Leaf Image")
            analyze_btn = gr.Button("Analyze Leaf", variant="primary")
            detail_btn = gr.Button("View technical details", variant="secondary")
            pred_note = gr.Markdown(visible=False)
            
        with gr.Column():
            gr.Markdown("### Screening Results:")
            annotated_output = gr.Image(type="pil", label="Detection Output")

            prediction_output = gr.Markdown()
            rec_output = gr.Markdown()

            with gr.Column(visible=False) as technical_section:
                gr.Markdown("## Technical Details")
                confidence_output = gr.Textbox(label="Model Confidence (Leaf Location + Disease Status)")
                fairness_output = gr.Textbox(label="Image Reliability (%)")
                confidence_note = gr.Markdown()

    analyze_btn.click(
        fn=analyze_citrus_leaf,
        inputs=uploaded_file,
        outputs=[annotated_output, prediction_output, rec_output, confidence_output, fairness_output, confidence_note]
    )

    detail_btn.click(
        fn=lambda: gr.update(visible=True),
        outputs=technical_section
    )

if __name__ == "__main__":
    demo.launch(share=True)
