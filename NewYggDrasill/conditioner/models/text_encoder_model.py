import torch

from typing import Optional, Dict, List

from .models.clip_te_model import CLIPTextEncoderModel



class TextEncoderModel(CLIPTextEncoderModel):
        # transformer_encoder: 

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
        prompt: List[str],
        num_images_per_prompt: int = 1,
        clip_skip: Optional[int] = None,
        lora_scale: Optional[float] = None,
        prompt_2: Optional[List[str]] = None,
        **kwargs,
    ):  
        clip_output = self.model.get_clip_embeddings(
            prompt=prompt,
            prompt_2=prompt_2,
            clip_skip=clip_skip,
            lora_scale=lora_scale,
        )

        

        return
