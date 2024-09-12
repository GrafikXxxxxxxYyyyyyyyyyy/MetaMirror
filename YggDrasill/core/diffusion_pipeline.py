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
    ForwardDiffusionOutput
)



class DiffusionPipelineInput(BaseOutput):
    forward_input: ForwardDiffusionInput
    width: Optional[int] = None
    height: Optional[int] = None
    image: Optional[PipelineImageInput] = None
    generator: Optional[torch.Generator] = None
    mask_image: Optional[PipelineImageInput] = None



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
        conditions: Dict[str, Any] = {},
        image: Optional[torch.FloatTensor] = None,
        generator: Optional[torch.Generator] = None,
        mask_image: Optional[torch.FloatTensor] = None,
        masked_image: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):  
        image_preprocessor = VaePipeline()


        forward = ForwardDiffusion(
            model_path=diffuser.path,
            device=diffuser.device,
            dtype=diffuser.dtype,
            scheduler_name=diffuser.scheduler_name
        )
        
        forward_input.num_channels = diffuser.num_channels
        forward_input.width = width or diffuser.sample_size
        forward_input.height = height or diffuser.sample_size

        forward_output = forward(**forward_input)   


        less_noisy_sample = forward_output.noisy_sample
        for i, t in enumerate(forward_output.timesteps):
            less_noisy_sample = diffuser(
                timestep=t,
                noisy_sample=less_noisy_sample,
                mask_sample=mask_image,
                masked_sample=masked_image,
                **conditions,
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
                    new_noisy_sample = model.scheduler.add_noise(
                        initial_image, noise, torch.tensor([noise_timestep])
                    ) 

                less_noisy_sample = (1 - initial_mask) * new_noisy_sample + initial_mask * less_noisy_sample        


        return less_noisy_sample 