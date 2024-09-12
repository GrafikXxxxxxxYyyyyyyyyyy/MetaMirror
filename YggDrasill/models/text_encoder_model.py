import torch

from typing import Optional

from .models.clip_te_model import CLIPTextEncoderModel



class TextEncoderModel:
    clip_encoder: CLIPTextEncoderModel
    # t5_encoder: Optional[TransformerModel] = None

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        model_type: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        **kwargs,
    ) -> None:  
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
        # Инитим модель CLIP
        self.clip_encoder = CLIPTextEncoderModel(
            model_path=model_path,
            model_type=model_type,
            device=device,
            dtype=dtype,
        )

        return
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #



    def __call__(
        self,
        **kwargs,
    ):
        # (
        #     prompt_embeds_1, 
        #     prompt_embeds_2, 
        #     pooled_prompt_embeds
        # ) = self.clip_encoder(
        #     prompt=prompt,
        #     prompt_2=prompt_2,
        #     clip_skip=clip_skip,
        #     lora_scale=lora_scale,
        # )

        # # TODO: Добавить процедуру запутывания эмбеддингов
        # # <CODE HERE>

        # # TODO: Описать класс трансформер модели transformer_te_model.py
        # # t5_embeds = self.t5_encoder(
        # #     prompt=prompt,
        # #     clip_skip=clip_skip,
        # #     lora_scale=lora_scale,
        # # )

        return
    
