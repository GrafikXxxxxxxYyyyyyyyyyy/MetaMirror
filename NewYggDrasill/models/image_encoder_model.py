import torch

from typing import Optional, Dict, List
from transformers import CLIPVisionModelWithProjection



class ImageEncoderModel:
    image_encoder: CLIPVisionModelWithProjection

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        model_type: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        **kwargs,
    ):  
        pass


    
    def get_image_embeddings(
        self,
        **kwargs,
    ):  
        pass



