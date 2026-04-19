# from copy import Error
import base64
import openai
from openai import OpenAI
from pathlib import Path
# import requests
import os
import time
from dotenv import load_dotenv

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_gpt(img_path, final_prompt, client, output_folder):
    # Path to your art/painting image
    # base64_image = encode_image(img_path)

    response = client.images.edit(
        model="gpt-image-1.5", # from api docs for editing images
        image = [open(img_path, "rb")],
        prompt=final_prompt#PROMPT
    )

    original_name = Path(img_path).name
    save_path = Path(output_folder) / original_name

    image_b64 = response.data[0].b64_json
    image_bytes = base64.b64decode(image_b64)
    with open(save_path, "wb") as f:
      f.write(image_bytes)
    print(f"Successfully saved restoration to: {save_path}")



if __name__ == "__main__":
  TIER = 'gpt_plain'
  IMG_FOLDER_PATH = f"../test/"
  OUTPUT_DIR = f"../outputs/{TIER}/"
  os.makedirs(OUTPUT_DIR, exist_ok = True)

  PROMPT = """Act like an expert image restoration and enhancement editor.
                Your main goal is to perfectly remove the mask overlays and complete the image at the mask regions.
                Edit this image to fix dull colors and low quality.
                Improve brightness, clarity, and sharpness carefully.
                Reduce noise and blur without losing details.
                Restore natural colors and contrast.
                Keep image medium, underlying textures and colors consistent.
                Do not over-process or exaggerate.
                Final result should not have any damaged regions and not be overly bright (white) or overlay dark (black).
                Focus on image enhancement, IMAGE RESTORATION, and quality improvement."""
  
  load_dotenv()  # Load environment variables from .env file
  client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

  for img_fname in os.listdir(IMG_FOLDER_PATH):
    print(f'Output to process as {OUTPUT_DIR}{img_fname}')
    if os.path.exists(f'{OUTPUT_DIR}{img_fname}'):
      print(IMG_FOLDER_PATH + img_fname, 'Output already processed')
      continue
    try:
      generate_gpt(IMG_FOLDER_PATH + img_fname, PROMPT, client, OUTPUT_DIR)
      time.sleep(10)
    except openai.BadRequestError as e:
      print("Error from GPT", e)
      time.sleep(10)
      continue
    break
