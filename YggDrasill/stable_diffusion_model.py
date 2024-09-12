import torch

from typing import Optional, Union

from .core.diffusion_model import DiffusionModel
from .models.text_encoder_model import TextEncoderModel
from .pipelines.text_encoder_pipeline import TextEncoderPipelineOutput



class StableDiffusionModel:
    diffuser: DiffusionModel
    text_encoder: Optional[TextEncoderModel] = None
    # image_encoder: Optional[ImageEncoderModel] = None

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        is_latent_model: bool = True,
        model_type: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        scheduler_name: Optional[str] = None,
        **kwargs,
    ):
        self.diffuser = DiffusionModel(
            model_path=model_path,
            model_type=model_type,
            device=device,
            dtype=dtype,
            scheduler_name=scheduler_name,
            is_latent_model=is_latent_model,
        )

        self.text_encoder = TextEncoderModel(
            model_path=model_path,
            model_type=model_type,
            device=device,
            dtype=dtype,
        )

        # self.image_encoder = 
    


    def __call__(
        self,
        te_output: TextEncoderPipelineOutput,
        use_refiner: bool = False,
        guidance_scale: float = 5.0,
        **kwargs,
    ) -> dict:
        """
        Подготавливает нужную последовательность входных аргументов
        и обуславливающих значений, соответсвующих заданной модели диффузии
        """
        conditions = {}

        if use_refiner:
            self.switch_to_refiner()

        self.diffuser.do_cfg = te_output.do_cfg
        self.diffuser.guidance_scale = guidance_scale

        # Пока что промпты только для sdxl
        conditions['prompt_embeds'] = (
            te_output.clip_embeds_2
            if use_refiner else 
            torch.concat([te_output.clip_embeds_1, te_output.clip_embeds_2], dim=-1)
        )

        # if ie_output is not None:
        #     pass

        conditions['added_cond_kwargs'] = {

        }


        return conditions
        