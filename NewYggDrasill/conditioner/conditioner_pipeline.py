import torch 

from dataclasses import dataclass
from diffusers.utils import BaseOutput
from typing import List, Optional, Union, Dict, Any

from .pipelines.text_encoder_pipeline import (
    ModelKey,
    TextEncoderPipeline, 
    TextEncoderPipelineInput, 
    TextEncoderPipelineOutput
)
from .conditioner_model import ConditionerModel






@dataclass
class ConditionerPipelineInput(BaseOutput):
    te_input: Optional[TextEncoderPipelineInput] = None






@dataclass
class ConditionerPipelineOutput(BaseOutput):
    prompt_embeds: torch.FloatTensor 
    batch_size: int = 1
    do_cfg: bool = False
    cross_attention_kwargs: Optional[dict] = None
    image_embeds: Optional[torch.FloatTensor] = None
    prompt_embeds_2: Optional[torch.FloatTensor] = None
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None





class ConditionerPipeline(
    TextEncoderPipeline,
        # ImageEncoderPipeline
):
    model: Optional[ConditionerModel] = None

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    def __init__(
        self,
        model_key: Optional[ModelKey] = None,
        **kwargs,
    ):
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
        if model_key is not None:
            self.model = ConditionerModel(**model_key)
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #

    

    # ################################################################################################################ #
    def retrieve_external_conditions(
        self,
        te_input: TextEncoderPipelineInput,
            # ie_output: Optional[ImageEncoderPipelineOutput] = None,
        **kwargs
    ):
    # ################################################################################################################ #
        print(te_input)

        # Собираем текстовые и картиночные условия генерации
        te_output: Optional[TextEncoderPipelineOutput] = None
        if te_input is not None:
            te_output = self.encode_prompt(**te_input)


        ############################################################################
        # По идее тут ещё модель что-то должна делать, но я хз что
        ############################################################################
        

        # if ie_input is not None:
        #     pass
        

        return ConditionerPipelineOutput(
            do_cfg=te_output.do_cfg,
            batch_size=te_output.batch_size,
            prompt_embeds=te_output.prompt_embeds,
            prompt_embeds_2=te_output.prompt_embeds_2,
            pooled_prompt_embeds=te_output.pooled_prompt_embeds,
            cross_attention_kwargs=te_output.cross_attention_kwargs,
        )
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

        return self.retrieve_external_conditions(input.te_input)
    # ================================================================================================================ #


    





