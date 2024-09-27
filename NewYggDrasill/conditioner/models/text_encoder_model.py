import torch

from typing import Optional, Dict, List

from .models.clip_te_model import CLIPTextEncoderModel
from .models.transformer_te_model import TransformerTextEncoderModel



class TextEncoderModel(CLIPTextEncoderModel):
    transformer_encoder: Optional[TransformerTextEncoderModel] = None

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        model_type: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        **kwargs,
    ):  
        super().__init__(
            dtype=dtype,
            device=device,
            model_path=model_path,
            model_type=model_type,
        )

        self.model_path = model_path
        self.model_type = model_type or "sd15"



    def __call__(
        self,
        prompt: List[str],
        num_images_per_prompt: int = 1,
        clip_skip: Optional[int] = None,
        lora_scale: Optional[float] = None,
        prompt_2: Optional[List[str]] = None,
        **kwargs,
    ):  
        (
            prompt_embeds_1, 
            prompt_embeds_2, 
            pooled_prompt_embeds
        ) = self.get_clip_embeddings(
            prompt=prompt,
            prompt_2=prompt_2,
            clip_skip=clip_skip,
            lora_scale=lora_scale,   
        )

        # if self.model_type == "":
        #     pass
        
        return (prompt_embeds_1, prompt_embeds_2, pooled_prompt_embeds)
