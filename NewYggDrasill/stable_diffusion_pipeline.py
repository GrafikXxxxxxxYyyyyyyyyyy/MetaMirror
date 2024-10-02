import torch 

from typing import Optional
from dataclasses import dataclass
from diffusers.utils import BaseOutput
from diffusers.image_processor import PipelineImageInput

from .core.diffusion_pipeline import DiffusionPipeline, DiffusionPipelineInput
from .stable_diffusion_model import StableDiffusionModel, StableDiffusionModelKey
from .pipelines.text_encoder_pipeline import TextEncoderPipeline, TextEncoderPipelineInput
from .pipelines.image_encoder_pipeline import ImageEncoderPipeline, ImageEncoderPipelineInput



@dataclass
class StableDiffusionPipelineInput(BaseOutput):
    diffusion_input: DiffusionPipelineInput
    te_input: Optional[TextEncoderPipelineInput] = None
    ie_input: Optional[ImageEncoderPipelineInput] = None
    # # Логика рефайнера
    # use_refiner: bool = False



@dataclass
class StableDiffusionPipelineOutput(BaseOutput):
    images: torch.FloatTensor



class StableDiffusionPipeline(
    DiffusionPipeline,
    TextEncoderPipeline,
    ImageEncoderPipeline
):  
    model: Optional[StableDiffusionModel] = None

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    def __init__(
        self,
        model_key: Optional[StableDiffusionModelKey] = None,
        **kwargs,
    ):
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
        if model_key is not None:
            self.model = StableDiffusionModel(**model_key)
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #



    # ================================================================================================================ #
    def __call__(
        self,
        model: StableDiffusionModel,
        diffusion_input: DiffusionPipelineInput,
        width: Optional[int] = None,
        height: Optional[int] = None,
        image: Optional[PipelineImageInput] = None,
        generator: Optional[torch.Generator] = None,
        mask_image: Optional[PipelineImageInput] = None,
        te_input: Optional[TextEncoderPipelineInput] = None,
        ie_input: Optional[ImageEncoderPipelineInput] = None,
    ):  
    # ================================================================================================================ #
        self.model = model        

        # Препроцессим входные изображения
        processor_output = self.maybe_process_images_latents(
            width = width,
            height = height,
            image = image,
            generator = generator,
            mask_image = mask_image,
        )
        image = processor_output.image_latents

        # Учитываем возможные пользовательские размеры изображений
        if image is not None:
            width, height = image.shape[2:]
        else:
            width = width or self.model.sample_size
            height = height or self.model.sample_size
        
        # # TODO: Чет изменений так дохуища, что лучше кажется просто создать 
        # diffusion_input = DiffusionPipelineInput(
        #     width=,
        #     height=,
        #     batch_size=,
        #     do_cfg=,
        #     guidance_scale=,
        #     image=,
        #     mask_image=,
        #     masked_image=,
        #     conditions=,
        # )

        # # Учитываем изменения на вход диффузии
        # diffusion_input.width = width
        # diffusion_input.height = height
        # diffusion_input.generator = generator
        # diffusion_input.image = processor_output.image_latents
        # diffusion_input.mask_image = processor_output.mask_latents
        # diffusion_input.masked_image = processor_output.masked_image_latents



        if "2. Обработка входных условий":
            if te_input is not None:
                te_output = self.encode_prompt(**te_input)

            if ie_input is not None:
                ie_output = self.encode_image(**ie_input)

        print(te_output)

        # # Вызов модели
        # diffusion_condition = self.model.retrieve_conditions


        if "3. Запуск основного диффузионного процесса (возможно с рефайнером)":
            pass
        


        return StableDiffusionPipelineOutput
    # ================================================================================================================ #











