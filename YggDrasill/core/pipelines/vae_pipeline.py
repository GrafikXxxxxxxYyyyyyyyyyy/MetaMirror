import torch

from dataclasses import dataclass
from diffusers.utils import BaseOutput
from diffusers.image_processor import VaeImageProcessor
from diffusers.image_processor import PipelineImageInput
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from ..models.vae_model import VaeModel



class VaePipelineInput(BaseOutput):
    # Размеры нужны, чтобы преобрзовать к ним переданное изображение
    width: Optional[int] = None
    height: Optional[int] = None
    image: Optional[PipelineImageInput] = None
    latents: Optional[torch.FloatTensor] = None
    generator: Optional[torch.Generator] = None
    mask_image: Optional[PipelineImageInput] = None



class VaePipelineOutput(BaseOutput):
    images: Optional[torch.FloatTensor] = None
    mask_latents: Optional[torch.FloatTensor] = None
    image_latents: Optional[torch.FloatTensor] = None
    masked_image_latents: Optional[torch.FloatTensor] = None



class VaePipeline:
    """
    Данный класс предоставляет функционал для использования
    модели Vae, а именно кодирование набора полученных изображений
    и их масок и последующее декодирование латентных представлений
    """
    def __call__(
        self, 
        width: Optional[int] = None,
        height: Optional[int] = None,
        vae: Optional[VaeModel] = None,
        image: Optional[PipelineImageInput] = None,
        generator: Optional[torch.Generator] = None,
        mask_image: Optional[PipelineImageInput] = None,
        latents: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> VaePipelineOutput:  
        """
        """
        use_vae = True if vae is not None else False

        self.image_processor = VaeImageProcessor(
            vae_scale_factor=(
                vae.scale_factor
                if use_vae else
                1
            )
        )
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=(
                vae.scale_factor
                if use_vae else
                1
            ), 
            do_normalize=False, 
            do_binarize=True, 
            do_convert_grayscale=True,
        )
        output = VaePipelineOutput()

        if image is not None:
            image = self.image_processor.preprocess(image)    
            output.image_latents = image
            if mask_image is not None:
                mask_image = self.mask_processor.preprocess(mask_image)        
                output.mask_latents = mask_image

        if use_vae:
            if latents is not None:
                images = vae.decode(latents)
                output.images = self.image_processor.postprocess(images.detach())
        
            if image is not None:
                image = image.to(
                    device=vae.device, 
                    dtype=vae.dtype
                )
                if (
                    height is not None 
                    and width is not None
                ):
                    # resize if sizes provided
                    image = torch.nn.functional.interpolate(
                        image, 
                        size=(height, width)
                    )
                output.image_latents = vae.encode(
                    image, 
                    generator    
                )

                if mask_image is not None:
                    mask_image = mask_image.to(
                        device=vae.device, 
                        dtype=vae.dtype
                    )
                    if (
                        height is not None 
                        and width is not None
                    ):
                        # resize if sizes provided
                        mask_image = torch.nn.functional.interpolate(
                            mask_image, 
                            size=(height, width)
                        )
                    masked_image = image * (mask_image < 0.5)
                    output.masked_image_latents = vae.encode(
                        masked_image, 
                        generator
                    )
                    output.mask_latents = torch.nn.functional.interpolate(
                        mask_image, 
                        size=(
                            height // vae.scale_factor, 
                            width // vae.scale_factor
                        )
                    )

        return output


