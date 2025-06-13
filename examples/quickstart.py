from difix.pipeline_difix import DifixPipeline
from diffusers.utils import load_image
from einops import rearrange
import numpy as np
import torch
import torchvision
from tqdm import tqdm


pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
pipe.set_progress_bar_config(disable=True)
pipe.to("cuda")

input_image = load_image("../../Difix3D/assets/bush_input.jpg")
input_image = input_image.crop((0, 0, 946, 532))
prompt = "remove degradation"

input_tensor = torch.tensor(np.array(input_image) / 255.0).cuda()
input_tensor = rearrange(input_tensor, "h w c -> 1 c h w")
print(input_tensor.shape)

output_image = pipe(prompt, image=input_tensor, num_inference_steps=1, timesteps=[199], guidance_scale=0.0, output_type="pt").images
print(output_image.shape)
# output_image.save("bush_output.png")




# pipe = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
# pipe.to("cuda")

# input_image = load_image("assets/bush_input.jpg")
# ref_image = load_image("assets/bush_ref.png")
# input_image = input_image.crop((0,0,ref_image.size[0], ref_image.size[1]))
# prompt = "remove degradation"

# input_tensor = torch.tensor(np.array(input_image) / 255.0).cuda()
# ref_tensor = torch.tensor(np.array(ref_image) / 255.0).cuda()
# print(input_tensor.shape, ref_tensor.shape)
# input_tensor = rearrange(input_tensor, "h w c -> 1 c h w")
# ref_tensor = rearrange(ref_tensor, "h w c -> 1 c h w")

# output_image = pipe(prompt, image=input_tensor, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
# output_image.save("bush_output.png")

# # output_image = pipe(prompt, image=input_tensor, ref_image=ref_tensor, num_inference_steps=1, timesteps=[199], guidance_scale=0.0, output_type="pt").images[0]
# # output_image = torchvision.transforms.functional.resize(output_image, input_tensor.shape[-2:])
# # print(output_image.shape, output_image.min(), output_image.max())
