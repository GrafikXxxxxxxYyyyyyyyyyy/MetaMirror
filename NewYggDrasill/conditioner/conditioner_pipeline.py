import torch 

from dataclasses import dataclass
from diffusers.utils import BaseOutput
from typing import List, Optional, Union, Dict, Any

from .pipelines.text_encoder_pipeline import (
    TextEncoderPipeline, 
    TextEncoderPipelineInput, 
    TextEncoderPipelineOutput
)
from .conditioner_model import ConditionerModel






@dataclass
class ConditionerPipelineInput(BaseOutput):
    te_input: Optional[TextEncoderPipelineInput] = None
    ie_input = None






@dataclass
class ConditionerPipelineOutput(BaseOutput):
    prompt_embeds: torch.FloatTensor 
    batch_size: int = 1
    do_cfg: bool = False
    cross_attention_kwargs: Optional[dict] = None
    text_embeds: Optional[torch.FloatTensor] = None
    refiner_prompt_embeds: Optional[torch.FloatTensor] = None






class ConditionerPipeline(
    TextEncoderPipeline,
        # ImageEncoderPipeline
):
    model: Optional[ConditionerModel] = None

    # # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    # def __init__(
    #     self,
    #     model_key: Optional[StableDiffusionModelKey] = None,
    #     **kwargs,
    # ):
    # # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    #     if model_key is not None:
    #         self.model = ConditionerModel(**model_key)
    # # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #

    

    # ################################################################################################################ #
    def retrieve_external_conditions(
        self,
        te_input: TextEncoderPipelineInput,
            # ie_output: Optional[ImageEncoderPipelineOutput] = None,
            # use_refiner
    ):
    # ################################################################################################################ #
        print(te_input)

        # Собираем текстовые и картиночные условия генерации
        te_output: Optional[TextEncoderPipelineOutput] = None
        # if "1. Вызывам собственный энкодер":
        if te_input is not None:
            te_output = self.encode_prompt(**te_input)

            do_cfg = te_output.do_cfg
            batch_size = te_output.batch_size
            cross_attention_kwargs = te_output.cross_attention_kwargs

        print(te_output)

        # if "2. Вызываем собственную модельку":
        #     (
        #         prompt_embeds, 
        #         text_embeds, 
        #         refiner_prompt_embeds
        #     ) = self.model.get_external_conditions(
        #         **te_output
        #     )
        
        
        
        # return ConditionerPipelineOutput(
        #     do_cfg=do_cfg,
        #     batch_size=batch_size,
        #     text_embeds=text_embeds,
        #     prompt_embeds=prompt_embeds,
        #     refiner_prompt_embeds=refiner_prompt_embeds,
        #     cross_attention_kwargs=cross_attention_kwargs,
        # )        
    # ################################################################################################################ #



    # ================================================================================================================ #
    def __call__(
        self,
        input: ConditionerPipelineInput, 
        conditioner: Optional[ConditionerModel] = None,
        **kwargs,
    ) -> ConditionerPipelineOutput:
    # ================================================================================================================ #
        if (
            conditioner is not None 
            and isinstance(conditioner, ConditionerModel)
        ):
            self.model = conditioner

        print(input)

        return self.retrieve_external_conditions(**input)
    # ================================================================================================================ #


    





