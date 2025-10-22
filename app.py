import gradio as gr
import os
import cv2
import torch
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from diffusers import AutoPipelineForText2Image
import requests
from PIL import Image # We need this to convert image formats

print("--- (1/4) Server starting: Loading FaceAnalysis... ---")
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

print("--- (2/4) Server starting: Loading INSwapper... ---")
model_path = "inswapper_128.onnx"
if not os.path.exists(model_path):
    print(f"--- Downloading {model_path}... ---")
    url = "https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx"
    r = requests.get(url, allow_redirects=True)
    with open(model_path, 'wb') as f:
        f.write(r.content)

swapper = get_model(model_path, download=False, download_zip=False, providers=["CPUExecutionProvider"])

print("--- (3/4) Server starting: Loading Stable Diffusion... ---")
pipe = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    torch_dtype=torch.float16 # Keep float16 for lower RAM usage on free tier
)
pipe = pipe.to("cpu")

print("--- (4/4) All models loaded. Ready to launch Gradio. ---")

# This is our main function that Gradio will call
def generate_story_image(source_img_pil, prompt_text, negative_prompt_text):

    # Gradio gives a PIL image, Insightface needs OpenCV (cv2) format
    # 1. Convert PIL (RGB) to OpenCV (BGR)
    print("--- Job received. Converting source image... ---")
    source_img = cv2.cvtColor(np.array(source_img_pil), cv2.COLOR_RGB2BGR)

    source_faces = app.get(source_img)

    if not source_faces:
        print("--- ERROR: No face found in source image. ---")
        raise gr.Error("No face found in the source image! Please upload a clearer, forward-facing photo.")

    source_face = source_faces[0]

    # 2. Generate the target image
    print(f"--- Generating target image for prompt: '{prompt_text}'... (This will be slow!) ---")
    target_img_pil = pipe(
    prompt=prompt_text, 
    negative_prompt=negative_prompt_text, 
    num_inference_steps=50  # <-- ADD THIS
).images[0]

    # 3. Convert target PIL (RGB) to OpenCV (BGR)
    target_img = cv2.cvtColor(np.array(target_img_pil), cv2.COLOR_RGB2BGR)

    target_faces = app.get(target_img)

    if not target_faces:
        print("--- ERROR: No face found in generated image. ---")
        raise gr.Error("No face was found in the generated image. Try a different prompt (e.g., 'a portrait of a person')")

    target_face = target_faces[0]

    # 4. Perform the swap
    print("--- Swapping faces... ---")
    result_img = swapper.get(target_img, target_face, source_face, paste_back=True)

    # 5. Convert final image back to PIL (RGB) for Gradio
    print("--- Job complete. Returning result. ---")
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_img_rgb)

# --- This creates the web UI ---
demo = gr.Interface(
    fn=generate_story_image,
    inputs=[
        gr.Image(label="Your Face", type="pil"),
        gr.Textbox(label="Story Prompt", info="e.g., 'cinematic portrait photo of an astronaut on Mars, dramatic lighting'"),
        gr.Textbox(label="Negative Prompt", info="e.g., 'cartoon, drawing, sketch, blurry, low quality, text, watermark'") # <-- ADD THIS TEXTBOX
    ],
    outputs=gr.Image(label="Generated Story Image"),
    title="AI Storybook Creator",
    description="Upload a clear photo of your face and write prompts to see yourself in the scene."
)

# --- This launches the web server ---
demo.launch()