import torch 

from dataclasses import dataclass
from diffusers.utils import BaseOutput
from typing import Optional, Dict, Any
from diffusers.image_processor import PipelineImageInput

from .pipelines.vae_pipeline import VaePipeline
from .diffusion_model import DiffusionModel, Conditions
from .pipelines.forward_diffusion import (
    ForwardDiffusion,
    ForwardDiffusionInput,
)
from .pipelines.backward_diffusion import BackwardDiffusionInput


@dataclass
class DiffusionPipelineInput(BaseOutput):
    forward_input: ForwardDiffusionInput
    width: Optional[int] = None
    height: Optional[int] = None
    image: Optional[PipelineImageInput] = None
    generator: Optional[torch.Generator] = None
    mask_image: Optional[PipelineImageInput] = None


@dataclass
class DiffusionPipelineOutput(BaseOutput):
    images: torch.FloatTensor



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
        conditions: Optional[Conditions] = None,
        image: Optional[torch.FloatTensor] = None,
        generator: Optional[torch.Generator] = None,
        mask_image: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):  
        print("DiffusionPipeline --->")

        # 0. Устанавливаем константы
        width = width or diffuser.sample_size
        height = height or diffuser.sample_size

        # Инитим форвард пайплайн из ключа модели
        FORWARD = ForwardDiffusion(**diffuser.key)
        
        # Препроцессим входные изображения
        IMAGE_PROCESSOR = VaePipeline()
        processor_output = IMAGE_PROCESSOR(
            vae=diffuser.vae,
            width=width,
            height=height,
            image=image,
            generator=generator,
            mask_image=mask_image,
        )
        image = processor_output.image_latents
        
        # Получаем пайп для шага обратного процесса из самой модели
        BACKWARD = diffuser(
            mask_image=processor_output.mask_latents,
            masked_image=processor_output.masked_image_latents,
        )
        
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
            noisy_sample=forward_output.noisy_sample, 
        )
        for i, t in enumerate(forward_output.timesteps):
            backward_input.timestep = t
            backward_input = BACKWARD(
                diffuser.predictor,
                **backward_input
            )

            # # маскирование шума для не inpaint модели, можно вынести в BackwardDiffusion
            # if (
            #     mask_image is not None
            #     and masked_image is not None
            #     and not diffuser.predictor.is_inpainting_model
            # ):
            #     initial_mask, _ = (
            #         mask_image.chunk(2)
            #         if diffuser.do_cfg else
            #         (mask_image, None)
            #     )

            #     if i < len(forward_output.timesteps) - 1:
            #         noise_timestep = forward_output.timesteps[i + 1]
            #         new_noisy_sample = BACKWARD.scheduler.add_noise(
            #             image, noise, torch.tensor([noise_timestep])
            #         ) 

            #     backward_input.noisy_sample = (1 - initial_mask) * new_noisy_sample                     \
            #                                     + initial_mask * backward_input.noisy_sample        

        vae_output = IMAGE_PROCESSOR(
            vae=diffuser.vae,
            latents=backward_input.noisy_sample
        )


        return DiffusionPipelineOutput(
            images=vae_output.images,
        )