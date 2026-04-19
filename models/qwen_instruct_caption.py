import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import base64
from io import BytesIO
import os

from .utils import image_to_base64, get_mask_prompt

def setup_autocaption_model():
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    # Use bfloat16 for speed/memory efficiency on modern GPUs
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        revision="main",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code = True,
        force_download=False
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code = True)

    return model, processor

def get_img_captions(pil_img, mask_prompt, processor, model):
    b64_img = image_to_base64(pil_img)#pil to base64

    query = (
        f"""
        <task>
    You are an image caption agent describing only the human desired content of an image. Given a base64 encoded image, give the image caption describing the following list of features:
    1. (Photo)/(Painting material)
    2. Image layout
    3. Image colors
    PLEASE ANSWER DIRECTLY TO THE QUESTION, DO NOT ADDRESS THE QUESTION ITSELF. DO NOT LIST THINKING STEPS, ADDRESS THE TASK DIRECTLY WITH ITS CONSTRAINTS.
    </task>

    <transformation_rule>
        The output MUST be at least 3x longer than the input and must include 10 new
        technical keywords lighting, subject and materials of the image.
        Exclude all {mask_prompt} degradations related descriptions.
        Exclude proper nouns.
    </transformation_rule>
        """
    )
    
    # 2. Define the XML-Tagged Instruction
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": b64_img},
                {"type": "text", "text": query}
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # Generate the optimized text
    generated_ids = model.generate(**inputs, max_new_tokens=150,# do_sample =True,
                                  #  repetition_penalty = 1.2,
                                  #  temperature = 0.7,
                                   top_p = 0.9)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return output_text

if __name__ == "__main__":
    TIER = 'nb2_pro'
    OUTPUT_DIR = f"../outputs/{TIER}/"
    IMG_FOLDER_PATH = f"../test/"
    os.makedirs(OUTPUT_DIR, exist_ok = True)

    PROMPT = 'a wave pattern image'

    model, processor = setup_autocaption_model()
    for filename in os.listdir(IMG_FOLDER_PATH):
        print(f'Output to process as {OUTPUT_DIR}{filename}')
        if os.path.exists(f'{OUTPUT_DIR}{filename}'):
            print(filename, 'Output already processed')
        else:
            PROMPT_T = get_img_captions(Image.open(IMG_FOLDER_PATH+filename), 
                             PROMPT, processor, model)
            print(f'Caption for {filename}: {PROMPT_T}')
        break

