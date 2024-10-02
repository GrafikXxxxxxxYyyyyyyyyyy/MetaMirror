import torch 

from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass
from diffusers.utils import BaseOutput
from diffusers.image_processor import PipelineImageInput

from .diffusion_model import DiffusionModel
from .pipelines.vae_pipeline import VaePipeline
from .pipelines.forward_diffusion import ForwardDiffusion, ForwardDiffusionInput



@dataclass
class DiffusionPipelineInput(BaseOutput):
    width: int = 512
    height: int = 512
    batch_size: int = 1
    do_cfg: bool = False
    guidance_scale: float = 5.0
    conditions: Optional[Conditions] = None
    image: Optional[torch.FloatTensor] = None
    generator: Optional[torch.Generator] = None
    mask_image: Optional[torch.FloatTensor] = None
    masked_image: Optional[torch.FloatTensor] = None
    forward_input: Optional[ForwardDiffusionInput] = None
    


@dataclass
class DiffusionPipelineOutput(BaseOutput):
    images: torch.FloatTensor



class DiffusionPipeline(VaePipeline, ForwardDiffusion):
    """
    Данный класс служит для того, чтобы выполнять полностью проход
    прямого и обратного диффузионного процессов
    """
    model: Optional[DiffusionModel] = None

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    def __init__(
        self,
        **kwargs,
    ):
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
        pass
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #



    def diffusion_process(
        self,
        width: int = 512,
        height: int = 512,
        batch_size: int = 1,
        do_cfg: bool = False,
        guidance_scale: float = 5.0,
        conditions: Optional[Conditions] = None,
        image: Optional[torch.FloatTensor] = None,
        generator: Optional[torch.Generator] = None,
        mask_image: Optional[torch.FloatTensor] = None,
        masked_image: Optional[torch.FloatTensor] = None,
        forward_input: Optional[ForwardDiffusionInput] = None,
        **kwargs,
    ):  
        """
        Выполняет полностью ПРЯМОЙ + ОБРАТНЫЙ диффузионные процессы из заданных условий
        """
        # # TODO: Вынести это на уровень StableDiffusion
        # # Учитываем batch_size если он был изменен
        # if mask_image is not None:
        #     mask_image = mask_image.repeat(
        #         batch_size // mask_image.shape[0], 1, 1, 1
        #     )
        # if masked_image is not None:
        #     masked_image = masked_image.repeat(
        #         batch_size // masked_image.shape[0], 1, 1, 1
        #     )



        if "Выполняется ForwardDiffusion":
            forward_input.sample = image
            forward_input.generator = generator
            forward_input.dtype = self.model.dtype
            forward_input.device = self.model.device
            # forward_input.denoising_end = 
            # forward_input.denoising_start = 

            
            forward_output = self.forward_pass(
                shape=(
                    batch_size,
                    self.model.num_channels,
                    width,
                    height,
                ),
                **forward_input,
            )

            print(forward_output)



        for i, t in tqdm(enumerate(forward_output.timesteps)):
            # Учитываем что может быть inpaint модель
            if self.model.predictor.is_inpainting_model:
                noisy_sample = torch.cat([noisy_sample, mask_image, masked_image], dim=1)   
                
            noisy_sample = self.model.backward_step(
                timestep=t,
                noisy_sample=noisy_sample,
                # do_cfg=do_cfg,
                # guidance_scale=guidance_scale,
                # conditions=conditions,
                **backward_coditions
            )
            
            # TODO: Добавить обработку маски через image
            # в случае если модель не для inpainting

        
        # images, _ = processor_pipeline(
        #     latents=backward_input.noisy_sample,
        # )


        # return DiffusionPipelineOutput(
        #     images=images,
        # )
    


    # ================================================================================================================ #
    def __call__(
        self,
        # diffuser: DiffusionModel,
        # batch_size: int = 1,
        # do_cfg: bool = False,
        # guidance_scale: float = 5.0,
        # width: Optional[int] = None,
        # height: Optional[int] = None,
        # conditions: Optional[Conditions] = None,
        # image: Optional[torch.FloatTensor] = None,
        # generator: Optional[torch.Generator] = None,
        # mask_image: Optional[torch.FloatTensor] = None,
        # forward_input: Optional[ForwardDiffusionInput] = None,
        **kwargs,
    ):  
    # ================================================================================================================ #
        pass
    # ================================================================================================================ #