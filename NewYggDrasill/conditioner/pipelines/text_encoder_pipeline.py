import torch 

from dataclasses import dataclass
from diffusers.utils import BaseOutput
from typing import List, Optional, Union, Dict, Any

from ...core.models.models.noise_predictor import ModelKey
from ..models.text_encoder_model import TextEncoderModel






@dataclass
class TextEncoderPipelineInput(BaseOutput):
    num_images_per_prompt: int = 1
    clip_skip: Optional[int] = None
    lora_scale: Optional[float] = None
    prompt: Optional[Union[str, List[str]]] = None
    prompt_2: Optional[Union[str, List[str]]] = None
    negative_prompt: Optional[Union[str, List[str]]] = None
    negative_prompt_2: Optional[Union[str, List[str]]] = None






@dataclass
class TextEncoderPipelineOutput(BaseOutput):
    do_cfg: bool
    batch_size: int
    clip_embeds_1: torch.FloatTensor
    clip_embeds_2: Optional[torch.FloatTensor] = None
        # transformer_embeds: Optional[torch.FloatTensor] = None
    pooled_clip_embeds: Optional[torch.FloatTensor] = None
    cross_attention_kwargs: Optional[Dict[str, Any]] = None






# НИ ОТ ЧЕГО НЕ НАСЛЕДУЕТСЯ + ХРАНИТ СВОЙ TextEncoder
class TextEncoderPipeline:  
    model: Optional[TextEncoderModel] = None

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    def __init__(
        self,
        model_key: Optional[ModelKey] = None,
        **kwargs,
    ):
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
        if model_key is not None:
            self.model = TextEncoderModel(**model_key)
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #



    def encode_prompt(
        self,
        num_images_per_prompt: int = 1,
        clip_skip: Optional[int] = None,
        lora_scale: Optional[float] = None,
        prompt: Optional[Union[str, List[str]]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> TextEncoderPipelineOutput:
        if "1. Нормализуем входные промпты":
            # Устанавливаем метку do_cfg исходя из наличия негативного промпта
            do_cfg = True if negative_prompt is not None else False

            prompt = prompt or ""
            prompt = [prompt] if isinstance(prompt, str) else prompt
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2
            if do_cfg:
                negative_prompt = negative_prompt or ""
                negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
                if len(prompt) != len(negative_prompt):
                    # Если негатив промпт не совпал с обычным, тупо зануляем все негативы
                    negative_prompt = [""] * len(prompt)

                negative_prompt_2 = negative_prompt_2 or negative_prompt
                negative_prompt_2 = [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2

            batch_size = len(prompt) * num_images_per_prompt


        if "2. Кодируем входные промпты":
            prompt_embeds_1, prompt_embeds_2, pooled_prompt_embeds = self.model(

            )
            if do_cfg:
                pass
            
            print(prompt_embeds_1.shape, prompt_embeds_2.shape, pooled_prompt_embeds.shape)



        # return TextEncoderPipelineOutput(
        #     do_cfg=do_cfg,
        #     batch_size=batch_size,
        #     clip_embeds_1=clip_output.prompt_embeds_1,
        #     clip_embeds_2=clip_output.prompt_embeds_2,
        #     pooled_clip_embeds=clip_output.pooled_prompt_embeds,
        #     cross_attention_kwargs=(
        #         {"scale": lora_scale}
        #         if lora_scale is not None else
        #         None
        #     )
        # )

    
    
    # ================================================================================================================ #
    def __call__(
        self,
        text_encoder: Optional[TextEncoderModel] = None,
        **kwargs,
    ) -> TextEncoderPipelineOutput:
    # ================================================================================================================ #
        if (
            text_encoder is not None 
            and isinstance(text_encoder, TextEncoderModel)
        ):
            self.model = text_encoder

        return self.encode_prompt(**kwargs)
    # ================================================================================================================ #

    

