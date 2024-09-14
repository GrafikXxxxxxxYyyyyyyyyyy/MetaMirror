import torch 

from typing import List, Optional
from dataclasses import dataclass
from diffusers.utils import BaseOutput

from ..models.noise_scheduler import NoiseScheduler
from ..models.noise_predictor import NoisePredictor


@dataclass
class BackwardDiffusionInput(BaseOutput):
    timestep: int
    noisy_sample: torch.FloatTensor



class BackwardDiffusion(NoiseScheduler):
    do_cfg: bool = False
    conditions: dict = {}
    guidance_scale: float = 5.0
    mask_sample: Optional[torch.FloatTensor] = None
    masked_sample: Optional[torch.FloatTensor] = None
    
    def __call__(
        self,
        predictor: NoisePredictor,
        timestep: int, 
        noisy_sample: torch.FloatTensor,
        **kwargs,
    ) -> BackwardDiffusionInput:
        """
        Данный пайплайн выполняет один полный шаг снятия шума в диффузионном процессе
        """
        # Учитываем CFG
        model_input = (
            torch.cat([noisy_sample] * 2)
            if self.do_cfg else
            noisy_sample
        )   

        # Скейлит входы модели
        model_input = self.scheduler.scale_model_input(
            timestep=timestep,
            sample=model_input,
        )

        # Конкатит маску и маскированную картинку для inpaint модели
        if (
            predictor.is_inpainting_model
            and mask_sample is not None
            and masked_sample is not None
        ):
            model_input = torch.cat([model_input, mask_sample, masked_sample], dim=1)   

        # Получаем предсказание шума
        noise_predict = predictor(
            timestep=timestep,
            noisy_sample=model_input,
            **self.conditions
        )

        # Учитываем CFG
        if self.do_cfg:
            negative_noise_pred, noise_pred = noise_predict.chunk(2)
            noise_predict = self.guidance_scale * (noise_pred - negative_noise_pred) + negative_noise_pred

        # Делаем шаг расшумления изображения 
        less_noisy_sample = self.scheduler.step(
            timestep=timestep,
            sample=noisy_sample,
            model_output=noise_predict,
        )

        return BackwardDiffusionInput(
            timestep=timestep,
            noisy_sample=less_noisy_sample,
            mask_sample=mask_sample,
            masked_sample=masked_sample,
        )

