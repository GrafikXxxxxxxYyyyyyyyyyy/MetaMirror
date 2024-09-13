import torch

from typing import Optional
from dataclasses import dataclass

from .models.vae_model import VaeModel
from .models.noise_scheduler import NoiseScheduler
from .models.noise_predictor import ModelKey, NoisePredictor



# @dataclass
# class DiffusionModelConditions(BaseOutput):
#     class_labels: Optional[torch.Tensor] = None


@dataclass
class DiffusionModelKey(ModelKey):
    is_latent_model: bool = False
    scheduler_name: Optional[str] = None


class DiffusionModel:
    key: DiffusionModelKey
    predictor: NoisePredictor
    do_cfg: bool = False
    guidance_scale: float = 5.0
    vae: Optional[VaeModel] = None

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    def __init__(
        self,
        is_latent_model: bool = False,
        scheduler_name: Optional[str] = None,
        **kwargs,
    ):     
        self.key = DiffusionModelKey(
            scheduler_name=scheduler_name,
            **kwargs,
        )
        self.diffuser = NoisePredictor(**kwargs)
        self.vae = VaeModel(**kwargs) if is_latent_model else None

    @property
    def dtype(self):
        return self.diffuser.dtype

    @property
    def device(self):
        return self.diffuser.device

    @property
    def sample_size(self):
        return (
            self.predictor.config.sample_size * self.vae.scale_factor
            if self.vae is not None else
            self.predictor.config.sample_size    
        )
    
    @property
    def num_channels(self):
        return (
            self.vae.config.latent_channels
            if self.vae is not None else
            self.predictor.config.in_channels
        )
    
    def switch_to_refiner(self):
        pass
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #



    # ================================================================================================================ #
    def __call__(
        self,
        **kwargs,
    ) -> torch.FloatTensor:
    # ================================================================================================================ #
        return 
    # ================================================================================================================ #