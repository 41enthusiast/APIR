from models import qwen_instruct_caption, clip_mask_prompt, nb2_pro_vertex_restore, gpt15_restore
import os
import torch
from PIL import Image
from dotenv import load_dotenv
import argparse

if __name__ == "__main__":
    VLM_CHOICES = ['nb2_pro', 'gpt15_restore']
    argparser = argparse.ArgumentParser(description="Run the APIR pipeline on a set of images.")
    argparser.add_argument("--img_folder", type=str, default="test/", help="Path to the folder containing input images.")
    argparser.add_argument("--output_dir", type=str, default="outputs/nb2_pro", help="Directory to save the output images.")
    argparser.add_argument("--model_choice", choices=VLM_CHOICES, default="nb2_pro", help="Choice of vision-language model to use.")
    args = argparser.parse_args()
                           
    TIER = args.model_choice
    OUTPUT_DIR = f"{args.output_dir}/{TIER}/"
    IMG_FOLDER_PATH = f"{args.img_folder}"
    os.makedirs(OUTPUT_DIR, exist_ok = True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    load_dotenv()  # Load environment variables from .env file
    if TIER == 'nb2_pro':
        project_id = os.getenv("GCP_PROJECT_ID")
        proj_loc = os.getenv("GCP_LOCATION")
        client = nb2_pro_vertex_restore.initialize_vertex_with_popup(project_id, proj_loc)
        print("Vertex AI client initialized with user credentials!")
        print('Ready to generate content with Vertex AI...')
    elif TIER == 'gpt15_restore':
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        print("OpenAI client initialized with API key!")
        print('Ready to generate content with OpenAI API...')

    mp_model, mp_processor, prompt_templates = clip_mask_prompt.mask_prompt_model(device)
    ic_model, ic_processor = qwen_instruct_caption.setup_autocaption_model()

    for filename in os.listdir(IMG_FOLDER_PATH):
        print(f'Output to process as {OUTPUT_DIR}{filename}')
        if os.path.exists(f'{OUTPUT_DIR}{filename}'):
            print(filename, 'Output already processed')
        else:
            mask_prompt = clip_mask_prompt.get_mask_prompt(Image.open(IMG_FOLDER_PATH+filename), 
                                                           device, mp_model, mp_processor, prompt_templates)
            print(f'Mask prompt for {filename}: {mask_prompt}')
            img_caption = qwen_instruct_caption.get_img_captions(Image.open(IMG_FOLDER_PATH+filename), 
                             mask_prompt, ic_processor, ic_model)
            print(f'Caption for {filename}: {img_caption}')
            if TIER == 'nb2_pro':
                apir_prompt = nb2_pro_vertex_restore.adaptive_generate_modeltext_from_text(client, 
                                                                                        img_caption, mask_prompt)
                print(f'Nano Banana 2 Pro Adaptive prompt for {filename}: {apir_prompt}')
                nb2_pro_vertex_restore.generate(IMG_FOLDER_PATH+filename, 
                                                apir_prompt, client, OUTPUT_DIR+filename) 
                print(f'Nano Banana 2 Pro Image restoration completed for {filename} and saved to {OUTPUT_DIR+filename}')
            elif TIER == 'gpt15_restore':
                apir_prompt = gpt15_restore.get_optimized_restoration_prompt(mask_prompt, img_caption, client)
                print(f'GPT-1.5 Restoration prompt for {filename}: {apir_prompt}')
                gpt15_restore.generate_gpt(IMG_FOLDER_PATH+filename, 
                                           apir_prompt, client, OUTPUT_DIR)
        break