
################################################################################################################

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import os
import requests
import base64
import io
import torch
from PIL import Image
import openai
import logging
from uuid import uuid4
import re
from diffusers import StableDiffusionPipeline

# Setting up Flask app

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Retrieve API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
HF_API_KEY = os.getenv("HF_API_KEY") 

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key is not set. Please set it in the environment or directly in the code.")

openai.api_key = OPENAI_API_KEY

def sanitize_prompt(prompt: str) -> str:
    """Remove or neutralize sensitive words in the prompt."""
    # List of sensitive words to neutralize
    sensitive_words = ["submissive", "physically and mentally unable", "lackey", "scapegoat"]
    sanitized_prompt = prompt
    for word in sensitive_words:
        sanitized_prompt = re.sub(word, "timid" if word == "submissive" else "reserved", sanitized_prompt, flags=re.IGNORECASE)
    return sanitized_prompt

def openai_refine_prompt(prompt: str):
    """Use GPT-4 to rephrase a prompt to reduce chances of content policy violations."""
    sanitized_prompt = sanitize_prompt(prompt)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": f"Please rephrase the following prompt to ensure it generates only a single character in the image: {sanitized_prompt}"}
            ],
        )
        refined_prompt = response['choices'][0]['message']['content']
        return refined_prompt
    except Exception as e:
        logger.error(f"Error refining prompt with GPT-4: {e}")
        return sanitized_prompt  # Fallback to sanitized prompt if refinement fails

# Function to generate image with DALL-E 3
def dalle_3_image_gen(prompt: str, size="1024x1024"):
    try:
        dalle_response = requests.post(
            "https://api.openai.com/v1/images/generations",
            headers={"Authorization": f"Bearer {openai.api_key}"},
            json={
                "model": "dall-e-3",
                "prompt": prompt,
                "size": size,
                "n": 1,
                "response_format": "b64_json",
            },
        )
        
        if dalle_response.status_code != 200:
            logger.error(f"DALL-E 3 API Error: {dalle_response.status_code} - {dalle_response.json()}")
            return None
        
        dalle_data = dalle_response.json()
        b64_json = dalle_data["data"][0]["b64_json"]
        
        # Convert base64 to bytes
        img_data = base64.b64decode(b64_json)
        return img_data
    
    except Exception as e:
        logger.error(f"Error in DALL-E 3 image generation: {e}")
        return None

# Function to generate image with DALL-E 2
def dalle_2_image_gen(prompt: str, size="1024x1024"):
    try:
        dalle_response = requests.post(
            "https://api.openai.com/v1/images/generations",
            headers={"Authorization": f"Bearer {openai.api_key}"},
            json={
                "model": "dall-e-2",
                "prompt": prompt,
                "size": size,
                "n": 1,
                "response_format": "b64_json",
            },
        )
        
        if dalle_response.status_code != 200:
            logger.error(f"DALL-E 2 API Error: {dalle_response.status_code} - {dalle_response.json()}")
            return None
        
        dalle_data = dalle_response.json()
        b64_json = dalle_data["data"][0]["b64_json"]
        
        # Convert base64 to bytes
        img_data = base64.b64decode(b64_json)
        return img_data
    
    except Exception as e:
        logger.error(f"Error in DALL-E 2 image generation: {e}")
        return None

# Function to generate image with Stable Diffusion v2 via Hugging Face
def stable_diffusion_v2_image_gen(prompt: str):
    try:
        api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        data = {"inputs": prompt}
        
        response = requests.post(api_url, headers=headers, json=data)
        
        if response.status_code != 200:
            logger.error(f"SD v2 API Error: {response.status_code} - {response.text}")
            return None
        
        return response.content
    
    except Exception as e:
        logger.error(f"Error in Stable Diffusion v2 image generation: {e}")
        return None

# Function to save image to disk and generate a URL
def save_image_to_disk(image_bytes, image_name, model_name):
    try:
        generated_images_dir = os.path.join("static", "generated_images")
        os.makedirs(generated_images_dir, exist_ok=True)  # Ensure directory exists

        image = Image.open(io.BytesIO(image_bytes))
        file_path = os.path.join(generated_images_dir, f"{image_name}_{model_name}.jpg")
        image.save(file_path)
        # return f"http://44.222.15.73:5000/static/generated_images/{image_name}_{model_name}.jpg"
        return f"http://127.0.0.1:5000/static/generated_images/{image_name}_{model_name}.jpg"
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        return None

def get_style_prompt(description: str, style: str, size="1024x1024"):
    # Define prompts for each style to focus on generating a single character only
    style_prompts = {
        "Modern Cartoon": f"{description}. Draw a single character only and nothing else should be in the final image except the character itself, fully visible, in a modern cartoon style. Use bold outlines, exaggerated expressions, and vibrant colors. Keep the background plain or minimal. No other characters or elements unless specified in the description.",
        
        "Realistic Cartoon": f"{description}. Draw a single character only in a realistic cartoon style. Keep the background plain or minimal. No additional characters unless specified in the description.",
        
        "Modern Anime": f"{description}. Create only one fully visible character in a Japanese anime style. Use large expressive eyes, smooth lines, and vivid colors. No other characters unless specified in the description.",
        
        "Chibi Drawing": f"{description}. Draw a single chibi-style character, with a large head, small body, and exaggerated expressions. Keep the background plain or minimal. Only one character unless specified in the description.",
        
        "Unique Cartoon Art": f"{description}. Create a single unique cartoon character with exaggerated features and distinctive designs. No other characters unless specified in the description.",
        
        "Modern Disney": f"{description}. Draw a Disney-style character, with rounded shapes and expressive eyes. No other characters or background elements unless specified in the description.",
    }
    
    # Select the style prompt and refine it
    styled_prompt = style_prompts.get(style, description)
    refined_prompt = styled_prompt + " This should be a standalone, single character image only unless multiple personas are requested in the description."
    
    openai_prompt = f"""
         You excel at creating lifelike characters with expertise in.

         **Task**: Your task is to draw a single character based solely on the description provided.

         **Instructions**
         - Focus on essential features.
         - Black and white by default.
         - Also provide full body single image.
         -f"Focus on an expressive face, dynamic posture, and thick lines with minimal shading. Background should be simple and minimal."
         -"Draw a single character in a classic comic book style with bold outlines, exaggerated facial expressions, and dynamic poses. a simple background to keep focus on the character. Capture the energy and drama commonly seen in superhero comics."
         - remember all the above pints while creating image.
         -"Draw a comic-style character with [describe clothing, pose, and accessories]. The character should be positioned in a [describe setting or environment]. Emphasize [describe any specific motion, expression, or other key features]. The background should be simple with minimal details, just enough to establish the environment. Use bold outlines, dynamic posture, exaggerated expressions, and a color palette that enhances the action and energy of the scene."
         
       **Description**: {description}

    **Output**: One standalone character image based solely on the user description.
    There should be single characteristics withing single image. No extras. And never forget this mentioned line
    """
    
    # Generate image URLs from all three models
    image_urls = []
    image_name = str(uuid4())
    
    # 1. DALL-E 3
    dalle3_image_data = dalle_3_image_gen(openai_prompt, size)
    if dalle3_image_data:
        dalle3_image_url = save_image_to_disk(dalle3_image_data, image_name, "dalle3")
        if dalle3_image_url:
            image_urls.append({"model": "DALL-E 3", "url": dalle3_image_url})
    
    # 2. DALL-E 2
    dalle2_prompt = f"{description}. Single character in {style} style."
    dalle2_image_data = dalle_2_image_gen(dalle2_prompt, size)
    if dalle2_image_data:
        dalle2_image_url = save_image_to_disk(dalle2_image_data, image_name, "dalle2")
        if dalle2_image_url:
            image_urls.append({"model": "DALL-E 2", "url": dalle2_image_url})
    
    # 3. Stable Diffusion v2
    sd_v2_prompt = f"{description}. Single character in {style} style."
    sd_v2_image_data = stable_diffusion_v2_image_gen(sd_v2_prompt)
    if sd_v2_image_data:
        sd_v2_image_url = save_image_to_disk(sd_v2_image_data, image_name, "sd_v2")
        if sd_v2_image_url:
            image_urls.append({"model": "Stable Diffusion v2", "url": sd_v2_image_url})
    
    return image_urls if image_urls else None

# Route to generate images from all three models based on provided text and style
@cross_origin()
@app.route('/aipy/gen_multi_model_image', methods=['POST'])
def gen_multi_model_image():
    feature_data = request.get_json()
    description = feature_data['text']
    style = feature_data['style']
    resolution = feature_data.get('resolution', '1024x1024')
    
    generated_img_urls = get_style_prompt(description, style, size=resolution)
    
    if generated_img_urls:
        logger.debug("Images have been generated from multiple models and are being sent back to front-end")
        return jsonify({"generated_img_urls": generated_img_urls})
    else:
        return jsonify({"error": "Image generation failed due to content policy or other issue."}), 500

# Legacy route for backward compatibility
@cross_origin()
@app.route('/aipy/gen_comic_anime_image', methods=['POST'])
def gen_comic_anime_image():
    feature_data = request.get_json()
    description = feature_data['text']
    style = feature_data['style']
    resolution = feature_data.get('resolution', '1024x1024')
    
    # This now returns results from all three models
    generated_img_urls = get_style_prompt(description, style, size=resolution)
    
    if generated_img_urls:
        logger.debug("Comic anime images have been generated and are being sent back to front-end")
        # Extract just the URLs for backward compatibility
        urls_only = [item["url"] for item in generated_img_urls]
        return jsonify({"generated_img_urls": urls_only})
    else:
        return jsonify({"error": "Image generation failed due to content policy or other issue."}), 500


@cross_origin()
@app.route('/', methods=['GET'])
def gen_comic_anime_image():
    return "Api created succesfully"


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)