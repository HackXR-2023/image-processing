import image_slicer

import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import torch

# load model and scheduler
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")

source_url = # Enter

# example image
url = source_url['data']

response = requests.get(url)
low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
low_res_img = low_res_img.resize((128, 128))

prompt = source_url['prompt']

upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
upscaled_image.save(img)


image_slicer.slice('example-mithax.png', 4)
