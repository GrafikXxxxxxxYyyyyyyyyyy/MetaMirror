import torch 

from typing import Optional
from dataclasses import dataclass
from diffusers.utils import BaseOutput

from .core.diffusion_pipeline import DiffusionPipeline, DiffusionPipelineInput
from .conditioner.conditioner_pipeline import ConditionerPipeline, ConditionerPipelineInput
from .stable_diffusion_model import StableDiffusionModel, StableDiffusionModelKey, StableDiffusionConditions






@dataclass
class StableDiffusionPipelineInput(BaseOutput):
    diffusion_input: DiffusionPipelineInput
    # use_refiner: bool = False
    # aesthetic_score: float = 6.0
    # negative_aesthetic_score: float = 2.5
    conditioner_input: Optional[ConditionerPipelineInput] = None






@dataclass
class StableDiffusionPipelineOutput(BaseOutput):
    images: torch.FloatTensor






class StableDiffusionPipeline(ConditionerPipeline, DiffusionPipeline):  

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
    ):  
    # ================================================================================================================ #
        self.model = model

        


        return StableDiffusionPipelineOutput)  
    # ================================================================================================================ #











