import os
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google import genai
from dotenv import load_dotenv
# import mimetypes
from google.genai import types
# import time
from google.genai import errors
import base64
from PIL import Image
from io import BytesIO

SCOPES = ['https://www.googleapis.com/auth/cloud-platform']

def auth_local_usr():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                '../client_secrets.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

def initialize_vertex_with_popup(project_id, project_location):
    user_credentials = auth_local_usr()
    print("Successfully authenticated!")
    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=project_location,
        credentials=user_credentials # Injecting the OAuth token here
    )
    return client

def adaptive_generate_modeltext_from_text(client, caption, degradation_class, model_name = 'gemini-2.5-flash'):
    prompt_text = f"""
    Refine this into a high-fidelity image restoration prompt:
    - Image Content: {caption}
    - Mask Issue to Resolve: {degradation_class}

    Instruction: Generate a prompt that describes the image restoration task,
    the type of mask deteriorating the image and
    describing the image content in bullet points.
    """
    
    response = client.models.generate_content(
        model=model_name, # High speed for prompt refinement
        contents=prompt_text
    )
    return response.text

def save_binary_file(file_name, data):
    f = open(file_name, "wb")
    f.write(data)
    f.close()
    print(f"File saved to to: {file_name}")

def generate(img_filename, final_prompt, client, output_filename):
    model = "gemini-3-pro-image-preview"#"gemini-3.1-flash-image-preview"#gemini-3-pro-image-preview
    
    # Create the proper Part object for gemini
    # image_bytes = base64.b64decode(b64_img.split(",")[1])
    with open(img_filename, "rb") as f:
        image_bytes = f.read()
    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type="image/jpeg"
    )

    contents = [
        types.Content(
            role="user",
            parts=[
                image_part,
                types.Part.from_text(text=final_prompt),
            ],
        ),
    ]
    tools = [
        types.Tool(googleSearch=types.GoogleSearch(
            search_types=types.SearchTypes(
                web_search=types.WebSearch(),
            ),
        )),
    ]
    generate_content_config = types.GenerateContentConfig(
        image_config = types.ImageConfig(
            aspect_ratio="1:1",
            image_size="1K",#imp for gemini 3 pro image preview, can be 1K or 2K, 2K sometimes causes OOM errors
        ),
        response_modalities=[
            "IMAGE",
            # "TEXT",
        ],
        tools=tools,
    )
    # file_index = 0
    try:
        #sync calls used instead because async sometimes fails
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )
        if response.parts:
            for part in response.parts:
                if part.inline_data and part.inline_data.data:
                    save_binary_file(output_filename, part.inline_data.data)
                    return # Success!
                elif part.text:
                    print("Model returned text instead of image:", part.text)
        else:
            print("Response succeeded but returned no parts.")

    #The Diagnostic Catcher
    except errors.APIError as e:
        print("\n" + "="*50)
        print("VERTEX AI REJECTION ERROR DETAILS")
        print(f"Status Code: {e.code}")
        print(f"Exact Reason: {e.message}")
        print(f"Full Details: {e.details}")
        print("="*50 + "\n")


if __name__ == "__main__":
    TIER = 'nb2_pro'
    OUTPUT_DIR = f"../outputs/{TIER}/"
    IMG_FOLDER_PATH = f"../test/"
    os.makedirs(OUTPUT_DIR, exist_ok = True)

    PROMPT = f"""Act like an expert image restoration and enhancement editor.
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
    project_id = os.getenv("GCP_PROJECT_ID")
    proj_loc = os.getenv("GCP_LOCATION")
    client = initialize_vertex_with_popup(project_id, proj_loc)
    print("Vertex AI client initialized with user credentials!")
    print('Ready to generate content with Vertex AI...')

    for filename in os.listdir(IMG_FOLDER_PATH):
        print(f'Output to process as {OUTPUT_DIR}{filename}')
        if os.path.exists(f'{OUTPUT_DIR}{filename}'):
            print(filename, 'Output already processed')
        else:
            generate(IMG_FOLDER_PATH+filename, PROMPT, client, OUTPUT_DIR+filename)
        # break