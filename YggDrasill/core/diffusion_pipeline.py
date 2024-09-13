import torch 

from dataclasses import dataclass
from diffusers.utils import BaseOutput
from typing import List, Optional, Dict, Any
from diffusers.image_processor import PipelineImageInput

from .diffusion_model import DiffusionModel
from .pipelines.vae_pipeline import VaePipeline
from .pipelines.forward_diffusion import (
    ForwardDiffusion,
    ForwardDiffusionInput,
)
from .pipelines.backward_diffusion import (
    BackwardDiffusion,
    BackwardDiffusionInput
)



class DiffusionPipelineInput(BaseOutput):
    forward_input: ForwardDiffusionInput
    width: Optional[int] = None
    height: Optional[int] = None
    image: Optional[PipelineImageInput] = None
    generator: Optional[torch.Generator] = None
    conditions: Optional[Dict[str, Any]] = None
    mask_image: Optional[PipelineImageInput] = None
    masked_image: Optional[torch.FloatTensor] = None



class DiffusionPipeline:
    """
    Данный класс служит для того, чтобы выполнять полностью проход
    прямого и обратного диффузионного процессов и учитывать использование VAE
    """
    def __call__(
        self,
        diffuser: DiffusionModel,
        forward_input: ForwardDiffusionInput,
        width: Optional[int] = None,
        height: Optional[int] = None,
        image: Optional[torch.FloatTensor] = None,
        generator: Optional[torch.Generator] = None,
        conditions: Optional[Dict[str, Any]] = None,
        mask_image: Optional[torch.FloatTensor] = None,
        masked_image: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):  
        print("DiffusionPipeline --->")
        IMAGE_PROCESSOR = VaePipeline()


        FORWARD = ForwardDiffusion(**diffuser.key)
        BACKWARD = BackwardDiffusion(**diffuser.key)
        BACKWARD.conditions = conditions
        BACKWARD.do_cfg = diffuser.do_cfg
        BACKWARD.guidance_scale = diffuser.guidance_scale
        
        forward_input.sample = image
        forward_input.generator = generator
        forward_input.num_channels = diffuser.num_channels
        forward_output = FORWARD(
            width=width or diffuser.sample_size,
            height=height or diffuser.sample_size,
            **forward_input
        )   


        backward_input = BackwardDiffusionInput(
            timestep=-1,
            mask_sample=mask_image,
            masked_sample=masked_image,
            noisy_sample=forward_output.noisy_sample, 
        )
        for i, t in enumerate(forward_output.timesteps):
            backward_input.timestep = t
            backward_input = BACKWARD(
                diffuser.predictor,
                **backward_input
            )

            # маскирование шума для не inpaint модели, можно вынести в BackwardDiffusion
            if (
                mask_image is not None
                and masked_image is not None
                and not diffuser.predictor.is_inpainting_model
            ):
                initial_mask, _ = (
                    mask_image.chunk(2)
                    if diffuser.do_cfg else
                    (mask_image, None)
                )

                if i < len(forward_output.timesteps) - 1:
                    noise_timestep = forward_output.timesteps[i + 1]
                    new_noisy_sample = BACKWARD.scheduler.add_noise(
                        image, noise, torch.tensor([noise_timestep])
                    ) 

                backward_input.noisy_sample = (1 - initial_mask) * new_noisy_sample                     \
                                                + initial_mask * backward_input.noisy_sample        


        return less_noisy_sample 