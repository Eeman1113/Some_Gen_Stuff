#___________________________________________________________________________________________________________________________

import streamlit as st 
import os

#___________________________________________________________________________________________________________________________

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from datasets import load_dataset
from PIL import Image  
import re

#___________________________________________________________________________________________________________________________

st.title('IMGTEXTA')

#___________________________________________________________________________________________________________________________

model_id = "CompVis/stable-diffusion-v1-4"
device = "cpu"

#___________________________________________________________________________________________________________________________

pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=st.secrets["AUTH_KEY"], torch_dtype=torch.float32)
def dummy(images, **kwargs): return images, False 
pipe.safety_checker = dummy

#___________________________________________________________________________________________________________________________

def infer(prompt, width, height, steps, scale, seed):      
    if seed == -1:	
        images_list = pipe(
            [prompt],
            height=height, 
            width=width,
            num_inference_steps=steps,
            guidance_scale=scale,
            generator=torch.Generator(device=device).manual_seed(seed))
    else:
        images_list = pipe(
            [prompt],
            height=height, 
            width=width,
            num_inference_steps=steps,
            guidance_scale=scale)

    return images_list["sample"]

#___________________________________________________________________________________________________________________________

def onclick(prompt):
    st.image(infer(prompt,512,512,30,7.5,-1))
prompt=st.text_input('Enter Your Prompt')
if prompt==True:
    onclick(prompt)

#___________________________________________________________________________________________________________________________

