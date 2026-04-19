import predict_mask
from transformers import CLIPProcessor, CLIPModel
import os
import torch
from PIL import Image

def mask_prompt_model(device):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").to(device)
    mask_classes = ["banded",
                "blotchy",
                "braided",
                "bubbly",
                "bumpy",
                "chequered",
                "cobwebbed",
                "cracked",
                "crosshatched",
                "crystalline",
                "dotted",
                "fibrous",
                "flecked",
                "freckled",
                "frilly",
                "gauzy",
                "grid",
                "grooved",
                "honeycombed",
                "interlaced",
                "knitted",
                "lacelike",
                "lined",
                "marbled",
                "matted",
                "meshed",
                "paisley",
                "perforated",
                "pitted",
                "pleated",
                "polka-dotted",
                "porous",
                "potholed",
                "scaly",
                "smeared",
                "spiralled",
                "sprinkled",
                "stained",
                "stratified",
                "striped",
                "studded",
                "swirly",
                "veined",
                "waffled",
                "woven",
                "wrinkled",
                "zigzagged"]
    prompt_template = [f"an image of {mask_cls} pattern" for mask_cls in mask_classes]
    
    return model, processor, prompt_template

def get_mask_prompt(pil_img, device, model, processor, prompt_template):
    mask_pred = predict_mask.predict(pil_img, device)

    inputs = processor(text=prompt_template, images=mask_pred, return_tensors="pt", padding=True)#.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    top_prob, top_label = probs.topk(1)
    print(f"Predicted mask prompt: {prompt_template[top_label]} with confidence {top_prob.item():.4f}")
    return prompt_template[top_label]


if __name__ == "__main__":
    TIER = 'clip_unet_efb0'
    OUTPUT_DIR = f"../outputs/{TIER}/"
    IMG_FOLDER_PATH = f"../test/"
    os.makedirs(OUTPUT_DIR, exist_ok = True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor, prompt_templates = mask_prompt_model(device)
    for filename in os.listdir(IMG_FOLDER_PATH):
        print(f'Output to process as {OUTPUT_DIR}{filename}')
        if os.path.exists(f'{OUTPUT_DIR}{filename}'):
            print(filename, 'Output already processed')
        else:
            mask_prompt = get_mask_prompt(Image.open(IMG_FOLDER_PATH+filename), 
                                          device, model, processor, prompt_templates)
            print(f'Mask prompt for {filename}: {mask_prompt}')
        break
