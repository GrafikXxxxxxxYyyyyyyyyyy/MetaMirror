import torch 

from dataclasses import dataclass
from diffusers.utils import BaseOutput
from typing import List, Optional, Union, Dict, Any

from ..core.diffusion_model import DiffusionModelKey
from ..models.image_encoder_model import ImageEncoderModel



@dataclass
class ImageEncoderPipelineInput(BaseOutput):
    pass



@dataclass
class ImageEncoderPipelineOutput(BaseOutput):
    pass



class ImageEncoderPipeline:  
    model: Optional[ImageEncoderModel] = None

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    def __init__(
        self,
        model_key: Optional[DiffusionModelKey] = None,
        **kwargs,
    ):
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
        if model_key is not None:
            self.model = ImageEncoderModel(**model_key)
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #



    def encode_image(
        self,
        **kwargs,
    ) -> ImageEncoderPipelineOutput:
        pass

    
    
    # ================================================================================================================ #
    def __call__(
        self,
        image_encoder: Optional[ImageEncoderModel] = None,
        **kwargs,
    ) -> ImageEncoderPipelineOutput:
    # ================================================================================================================ #
        if (
            image_encoder is not None 
            and isinstance(image_encoder, ImageEncoderModel)
        ):
            self.model = image_encoder

        return self.encode_image(**kwargs)
    # ================================================================================================================ #

    

