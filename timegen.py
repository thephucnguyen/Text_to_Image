import os

import argparse
from random import getrandbits
import numpy as np
from einops import rearrange
import cv2
import time

from .triton.triton_request_util import init_triton_client
from .models.clip import ClipModel
from .models.unet import UnetModel
from .models.vae import AutoEncoder
from .models.upscale import UpscaleModel
from .scheduler.plms import PLMS


class Config():
    def __init__(self):
        self.verbose = False
        self.url = 'localhost:8003'
        self.protocol = "HTTP"

        self.model_name_text_encoder = 'text_encoder'
        self.model_name_unet = 'unet'
        self.model_name_unet_input_block = 'unet_input_block'
        self.model_name_unet_middle_block = 'unet_middle_block'
        self.model_name_unet_output_block = 'unet_output_block'
        self.model_name_vae_decoder = 'vae_decoder'
        self.model_name_scale_image = 'scale'

        self.is_split_unet = False
        self.tokenizer = 'openai/clip-vit-large-patch14'


def init_inference_config():

    infer_config = Config()

    protocol = infer_config.protocol.lower()

    # init triton client
    triton_client = init_triton_client(
        protocol, infer_config.url, infer_config.verbose)

    return infer_config, triton_client, protocol


TYPE = np.float16
infer_config, triton_client, protocol = init_inference_config()

# Clip encoder
model_name_text_encoder = infer_config.model_name_text_encoder
text_encoder = ClipModel(triton_client, protocol,
                         infer_config.tokenizer, model_name_text_encoder)

if not infer_config.is_split_unet:
    model_name_unet = infer_config.model_name_unet
    # Diffusion model
    diffusion_model = UnetModel(
        triton_client, protocol, model_name_unet, is_split_unet=infer_config.is_split_unet)

else:
    model_name_unet_input_block = infer_config.model_name_unet_input_block
    model_name_unet_middle_block = infer_config.model_name_unet_middle_block
    model_name_unet_output_block = infer_config.model_name_unet_output_block
    diffusion_model = UnetModel(triton_client, protocol, None, model_name_unet_input_block,
                                model_name_unet_middle_block, model_name_unet_output_block, infer_config.is_split_unet)

# Vae_decoder
model_name_vae_decoder = infer_config.model_name_vae_decoder
vae_decoder = AutoEncoder(triton_client, protocol, model_name_vae_decoder)

model_name_scale_image = infer_config.model_name_scale_image
try:
    up_scaler = UpscaleModel(triton_client, protocol, model_name_scale_image)
except:
    print('Not found upscaler model!')
    up_scaler = None 


def put_content(img, texts="TEIKI\nSHINKA NETWORK", 
                text_colors=[(255, 255, 255), (255, 255, 255)], 
                text_font=cv2.FONT_HERSHEY_DUPLEX, 
                thickness=2, 
                logos_path=[]):
    h,w,_ = img.shape
    list_logos = []
    for path in logos_path:
        logo = cv2.imread(path)
        logo_h, logo_w, _ = logo.shape
        logo = cv2.resize(logo,(int(w/512*28),int(h/512*28)))
        list_logos.append(logo)
    fontScale = w/512*1.3
    new_h = w*9/16
    new_min_y = int(h/2-new_h/2)
    new_max_y = int(new_min_y+new_h)
    img = img[new_min_y:new_max_y,:]
    lines = texts.split("\n")
    normalized_lines_pos = [(int(0.08*w), int(0.25*h)), (int(0.08*w), int(0.35*h))]  
    img = cv2.putText(img, lines[0], normalized_lines_pos[0], text_font, fontScale, text_colors[0], thickness, cv2.LINE_AA)
    img = cv2.putText(img, lines[1], normalized_lines_pos[1], text_font, fontScale, text_colors[1], thickness, cv2.LINE_AA)
    left_border_logo = int(0.08*w)
    for logo in list_logos:
        h_lg, w_lg, _ = logo.shape
        img[int(0.1*h):int(0.1*h)+h_lg, left_border_logo:left_border_logo+w_lg] = logo
        left_border_logo = left_border_logo+w_lg+int(0.02*w)
    return img

def Text2Img(id: str,
             prompt: str,
             batch_size: int = 1,
             n_iter: int = 2,
             seed: int = None,
             max_length: int = 77,
             size_out: tuple = (4, 512, 512),  # (c, h, w)
             factor: int = 8,  # downsampling factor
             ddim_steps: int = 50,
             ddim_eta: float = 0.0,
             scale: float = 7.5,
             scale_factor=0.18215,
             is_scale_up: bool = False,
             is_save_img: bool = False,
             output_dir: str = None,
             ) -> list:

    if seed is None:
        seed = getrandbits(32)
    # Global seed
    np.random.seed(seed)
    assert prompt is not None
    data = [batch_size * [prompt]]
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = './'

    shape = [size_out[0], size_out[1] // factor, size_out[2] // factor]
    start_code = None

    sampler = PLMS(diffusion_model, ddim_steps, batch_size,
                   shape, ddim_eta, verbose=False, dtype=TYPE)

    outputs = []
    for n in range(n_iter):
        start = time.time()
        for prompts in data:
            uc = None

            if scale != 1.0:
                uc = text_encoder.encode("", max_length)
            c = text_encoder.encode(prompts, max_length)
            # uc = np.zeros((1, 77, 768), dtype=TYPE)
            # c = np.zeros((1, 77, 768), dtype=TYPE)
            samples_ddim, _ = sampler.sample(batch_size=batch_size,
                                             conditioning=c,
                                             x_T=start_code,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             )
            start1 = time.time()
            samples_ddim = 1 / scale_factor * samples_ddim
            print(samples_ddim.dtype)
            x_samples_ddim = vae_decoder.decode(samples_ddim)
            print('Time of vae_decoder:', time.time() - start1)
            x_samples_ddim = np.clip(
                (x_samples_ddim + 1.0) / 2.0, a_min=0.0, a_max=1.0)

            # x_samples_ddim = np.transpose(x_samples_ddim, (0, 2, 3, 1))
            # x_samples_ddim = np.transpose(
            #     x_samples_ddim, (0, 3, 1, 2)).astype(typetrt2np(dtype_scale_image))
            # print(x_samples_ddim.shape)
            if is_scale_up and up_scaler is not None:
                x_samples_ddim = up_scaler.transform(x_samples_ddim)
                x_samples_ddim = np.clip(x_samples_ddim, a_min=0.0, a_max=1.0)

            for i, x_sample in enumerate(x_samples_ddim):
                x_sample = 255. * rearrange(x_sample, 'c h w -> h w c')
                if is_save_img:
                    out = cv2.cvtColor(x_sample.astype(
                        np.uint8), cv2.COLOR_RGB2BGR)
                    out = put_content(out)
                    output_path = os.path.join(output_dir, f'{id}_{n}_{i}.png')
                    outputs.append(output_path)
                    cv2.imwrite(output_path, out)
        print(f'Time inference n_iter {n}: ', time.time() - start)

    return outputs

# def upscale_img(img):


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--prompt", type=str, required=False,
                        nargs="?", default='a dog wear a funny hat', help="Input sentence.")

    parser.add_argument("--batch_size", type=int, required=False,
                        nargs="?", default=1, help="Batch size.")
    parser.add_argument("--output_dir", type=str,
                        required=False, default='./tmp', help="Output path to save image")
    parser.add_argument("--n_iter", type=int,
                        required=False, default=2, help="Number of generation images")

    parser.add_argument("--seed", type=int,
                        required=False, default=None, help="Ransom seed")
    parser.add_argument("--split_unet", action='store_true',
                        help="Is Unet split to input, middle, output block?")
    parser.add_argument("--up_scale", action='store_true')

    opt = parser.parse_args()

    outputs = Text2Img('abc',
                       opt.prompt,
                       batch_size=opt.batch_size,
                       n_iter=opt.n_iter,
                       seed=opt.seed,
                       output_dir=opt.output_dir,
                       is_split_unet=opt.split_unet, is_scale_up=opt.up_scale, is_save_img=True)

    print('Save imgs at: ', outputs)
