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
    use_refiner: bool = False
    guidance_scale: float = 5.0
    num_images_per_prompt: int= 1
    te_input: Optional[TextEncoderPipelineInput] = None
    # ie_input: Optional[ImageEncoderPipelineInput] = None
    pass


@dataclass
class StableDiffusionPipelineOutput(BaseOutput):
    images: torch.FloatTensor



class StableDiffusionPipeline:
    def __call__(
        self,
        model: StableDiffusionModel,
        diffusion_input: DiffusionPipelineInput,
        te_input: Optional[TextEncoderPipelineInput] = None,
        use_refiner: bool = False,
        guidance_scale: float = 5.0,
        num_images_per_prompt: int = 1,
                # ip_adapter_image: Optional[PipelineImageInput] = None,
                # output_type: str = "pt",
        refiner_steps: Optional[int] = None,
        refiner_scale: Optional[float] = None,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
        **kwargs,
    ):
        """
        """
        print("StableDiffusionPipeline --->")
        
        if te_input is not None and model.text_encoder is not None:
            te_pipeline = TextEncoderPipeline()
            te_output = te_pipeline(
                model.text_encoder,
                **te_input,
            )

        # Вызов модели просто формирует словарь условий и сохраняет, 
        # которsq будет использована
        # Дальше эти условия передаются в пайплайн для расшумления
        conditions = model(
            te_output=te_output,
            use_refiner=use_refiner,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            aesthetic_score=aesthetic_score,
            negative_aesthetic_score=negative_aesthetic_score,
        )
            
        diffusion_pipeline = DiffusionPipeline()


        if use_refiner:
        #     # if (
        #     #     refiner_scale is not None
        #     #     and isinstance(refiner_scale, float)
        #     #     and refiner_scale < 1.0
        #     #     and refiner_scale > 0.0
        #     # ):
            pass
        

        images = diffusion_pipeline(
            model.diffuser,
            **diffusion_input,
        )

        return images








































        # if "Вызываем основной диффузионный пайп":
        #     diffusion = DiffusionPipeline()

            # if (
            #     refiner_scale is not None
            #     and isinstance(refiner_scale, float)
            #     and refiner_scale < 1.0
            #     and refiner_scale > 0.0
            # ):
        #         # Первый шаг 
        #         diffusion_input.forward_input.denoising_end = refiner_scale        
        #         diffusion_output = diffusion(
        #             model.diffuser,
        #             **diffusion_input,
        #         )

        #         # Вызывается модель, чтобы переключиться на рефайнер
        #         _ = model(
        #             te_output,
        #             use_refiner=use_refiner,
        #         )
                
        #         # Второй шаг
        #         diffusion_input.forward_input.denoising_end = None
        #         diffusion_input.forward_input.denoising_start = refiner_scale
        #         diffusion_output = diffusion(
        #             model.diffuser,
        #             **diffusion_input,
        #         )

            
        #     if (
        #         refiner_steps is not None
        #         and isinstance(refiner_steps, int)
        #         and refiner_scale > 0
        #     ):
        #         diffusion_input.forward_input.num_inference_steps = refiner_steps
                
        #     diffusion_output = diffusion(
        #         model.diffuser,
        #         **diffusion_input,
        #     )


        # return StableDiffusionPipelineOutput(
        #     clear_sample=output,
        # )