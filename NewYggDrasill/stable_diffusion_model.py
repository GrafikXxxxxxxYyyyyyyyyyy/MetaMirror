import torch

from typing import Optional
from dataclasses import dataclass

from .models.text_encoder_model import TextEncoderModel
from .models.image_encoder_model import ImageEncoderModel
from .pipelines.text_encoder_pipeline import TextEncoderPipelineOutput
from .core.diffusion_model import DiffusionModelKey, DiffusionModelConditions, DiffusionModel



@dataclass
class StableDiffusionModelKey(DiffusionModelKey):
    use_ip_adapter: bool = False
    use_text_encoder: bool = True



@dataclass
class StableDiffusionConditions(DiffusionModelConditions):
    use_refiner: bool = False



class StableDiffusionModel(
    DiffusionModel, 
    TextEncoderModel, 
    ImageEncoderModel
):  
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        use_text_encoder: bool = True,
        is_latent_model: bool = False,
        use_image_encoder: bool = False,
        model_type: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        scheduler_name: Optional[str] = None,
        **kwargs,
    ): 
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #    
        # Инитим диффузионную модель
        DiffusionModel.__init__(
            self,
            dtype=dtype,
            device=device,
            model_path=model_path,
            model_type=model_type,
            scheduler_name=scheduler_name,
            is_latent_model=is_latent_model,
        )

        # И возможно условную модель, если нужно обуславливание
        self.use_text_encoder = use_text_encoder
        if use_text_encoder:
            TextEncoderModel.__init__(
                self,
                dtype=dtype,
                device=device,
                model_path=model_path,
                model_type=model_type,
            )

        self.use_image_encoder = use_image_encoder
        if use_image_encoder:
            ImageEncoderModel.__init__(
                self,
                dtype=dtype,
                device=device,
                model_path=model_path,
                model_type=model_type,
            )

        print("\t<<<StableDiffusionModel ready!>>>\t")
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #    



    # ################################################################################################################ #
    def _get_add_time_ids(
        self,
        original_size,
        crops_coords_top_left,
        aesthetic_score,
        negative_aesthetic_score,
        target_size,
        negative_original_size,
        negative_crops_coords_top_left,
        negative_target_size,
        addition_time_embed_dim,
        expected_add_embed_dim,
        dtype,
        text_encoder_projection_dim,
        requires_aesthetics_score,
    ):
        if requires_aesthetics_score:
            add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
            add_neg_time_ids = list(
                negative_original_size + negative_crops_coords_top_left + (negative_aesthetic_score,)
            )
        else:
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_neg_time_ids = list(negative_original_size + crops_coords_top_left + negative_target_size)

        passed_add_embed_dim = (
            addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )

        if (
            expected_add_embed_dim < passed_add_embed_dim
            and (passed_add_embed_dim - expected_add_embed_dim) == addition_time_embed_dim
        ):
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to disable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=False)` to make sure `target_size` {target_size} is correctly used by the model."
            )
        elif expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_neg_time_ids = torch.tensor([add_neg_time_ids], dtype=dtype)

        return add_time_ids, add_neg_time_ids
    

    def retrieve_conditions(
        self,
        width: int,
        height: int,
        batch_size: int = 1,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
        te_output: Optional[TextEncoderPipelineOutput] = None,
        **kwargs,
    ):
        if self.model_type == "sd15":
            pass

        elif self.model_type == "sdxl":
            # Для модели SDXL почему-то нужно обязательно расширить 
            # дополнительные аргументы временными метками 
            add_time_ids, add_neg_time_ids = self._get_add_time_ids(
                original_size = (height, width),
                crops_coords_top_left = (0, 0),
                aesthetic_score = self.aesthetic_score,
                negative_aesthetic_score = self.negative_aesthetic_score,
                target_size = (height, width),
                negative_original_size = (height, width),
                negative_crops_coords_top_left = (0, 0),
                negative_target_size = (height, width),
                addition_time_embed_dim = self.predictor.config.addition_time_embed_dim,
                expected_add_embed_dim = self.add_embed_dim,
                dtype = self.model.dtype,
                text_encoder_projection_dim = self.text_encoder_projection_dim,
                requires_aesthetics_score = self.use_refiner,
            )
            add_time_ids = add_time_ids.repeat(batch_size, 1)
            add_neg_time_ids = add_neg_time_ids.repeat(batch_size, 1)
            
            if do_cfg:
                add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)
            
            conditions.added_cond_kwargs["time_ids"] = add_time_ids.to(self.model.device)

        elif self.model.model_type == "sd3":
            pass

        elif self.model.model_type == "flux":
            pass
    # ################################################################################################################ #



    # ================================================================================================================ #
    def __call__(self, **kwargs):
    # ================================================================================================================ #
        print("DiffusionModel --->")

        return self.get_diffusion_conditions(**kwargs)
    # ================================================================================================================ #
        