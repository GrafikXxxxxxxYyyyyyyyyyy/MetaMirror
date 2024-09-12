import torch

from typing import Optional, Union

from .models.vae_model import VaeModel
from .models.noise_predictor import NoisePredictor
from .models.noise_scheduler import NoiseScheduler



class DiffusionModel:
    predictor: NoisePredictor
    scheduler: NoiseScheduler
    do_cfg: bool = False
    guidance_scale: float = 5.0
    vae: Optional[VaeModel] = None

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        model_type: Optional[str] = None,
        is_latent_model: bool = False,
        dtype: torch.dtype = torch.float16,
        scheduler_name: Optional[str] = None,
        **kwargs,
    ):     
        self.predictor = NoisePredictor(
            model_path=model_path,
            model_type=model_type,
            device=device,
            dtype=dtype,
        )

        self.scheduler = NoiseScheduler(
            model_path=model_path,
            model_type=model_type,
            device=device,
            dtype=dtype,
            scheduler_name=scheduler_name,
        )

        self.vae = (
            VaeModel(
                model_path=model_path,
                model_type=model_type,
                device=device,
                dtype=dtype,
            )
            if is_latent_model else
            None
        )

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



    # ================================================================================================================ #
    def __call__(
        self,
        timestep: int, 
        noisy_sample: torch.FloatTensor,
        mask_sample: Optional[torch.FloatTensor] = None,
        masked_sample: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
    # ================================================================================================================ #
        """
            # Получает на вход временную метку и шумный вход
            # Поочерёдно выполняет прогон входного сэмпла черед планировщик и предиктор
        """
        # Удваивает шумовые сэмплы
        model_input = (
            torch.cat([noisy_sample] * 2)
            if self.do_cfg else
            noisy_sample
        )   

        model_input = self.scheduler(
            timestep=timestep,
            noisy_sample=model_input,
        )

        if (
            self.predictor.is_inpainting_model
            and mask_sample is not None
            and masked_sample is not None
        ):
            # Конкатит маску и маскированную картинку для inpaint модели
            model_input = torch.cat([model_input, mask_sample, masked_sample], dim=1)   

        noise_predict = self.predictor(
            timestep=timestep,
            noisy_sample=model_input,
            **kwargs
        )

        if self.do_cfg:
            negative_noise_pred, noise_pred = noise_predict.chunk(2)
            noise_predict = self.guidance_scale * (noise_pred - negative_noise_pred) + negative_noise_pred

        less_noisy_sample = self.scheduler(
            timestep=timestep,
            noisy_sample=noisy_sample,
            noise_predict=noise_predict,
        )

        return less_noisy_sample
    # ================================================================================================================ #