import torch 

from typing import List, Optional
from dataclasses import dataclass
from diffusers.utils import BaseOutput

from .stable_diffusion_model import StableDiffusionModel
from .core.diffusion_pipeline import (
    DiffusionPipeline,
    DiffusionPipelineInput,
)
from .pipelines.text_encoder_pipeline import (
    TextEncoderPipeline,
    TextEncoderPipelineInput,
    TextEncoderPipelineOutput
)


@dataclass
class StableDiffusionPipelineInput(BaseOutput):
    diffusion_input: DiffusionPipelineInput
    guidance_scale: float = 5.0
    num_images_per_prompt: int= 1
    te_input: Optional[TextEncoderPipelineInput] = None
    # ie_input: Optional[ImageEncoderPipelineInput] = None
    pass


@dataclass
class StableDiffusionPipelineOutput(BaseOutput):
    pass



class StableDiffusionPipeline:
    def __call__(
        self,
        model: StableDiffusionModel,
        diffusion_input: DiffusionPipelineInput,
        guidance_scale: float = 5.0,
        num_images_per_prompt: int = 1,
        te_input: Optional[TextEncoderPipelineInput] = None,
        # ip_adapter_image: Optional[PipelineImageInput] = None,
        # output_type: str = "pt",
        use_refiner: bool = False,
        refiner_steps: Optional[int] = None,
        refiner_scale: Optional[float] = None,
        # aesthetic_score: float = 6.0,
        # negative_aesthetic_score: float = 2.5,
        **kwargs,
    ):
        """
        """
        use_refiner = refiner_scale is not None or refiner_steps is not None
        
        if te_input is not None and model.text_encoder is not None:
            te_pipeline = TextEncoderPipeline()
            te_output = te_pipeline(
                model.text_encoder,
                **te_input,
            )

        conditions = model(
            te_output
        )
            

        if "Вызываем основной диффузионный пайп":
            diffusion = DiffusionPipeline()

            if (
                refiner_scale is not None
                and isinstance(refiner_scale, float)
                and refiner_scale < 1.0
                and refiner_scale > 0.0
            ):
                # Первый шаг 
                diffusion_input.forward_input.denoising_end = refiner_scale        
                diffusion_output = diffusion(
                    model.diffuser,
                    **diffusion_input,
                )

                # Вызывается модель, чтобы переключиться на рефайнер
                _ = model(
                    te_output,
                    use_refiner=use_refiner,
                )
                
                # Второй шаг
                diffusion_input.forward_input.denoising_end = None
                diffusion_input.forward_input.denoising_start = refiner_scale
                diffusion_output = diffusion(
                    model.diffuser,
                    **diffusion_input,
                )

            
            if (
                refiner_steps is not None
                and isinstance(refiner_steps, int)
                and refiner_scale > 0
            ):
                diffusion_input.forward_input.num_inference_steps = refiner_steps
                
            diffusion_output = diffusion(
                model.diffuser,
                **diffusion_input,
            )


        return StableDiffusionPipelineOutput(
            clear_sample=output,
        )