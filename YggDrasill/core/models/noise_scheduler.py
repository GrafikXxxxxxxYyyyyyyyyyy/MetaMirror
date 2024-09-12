import torch

from diffusers import (
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)
from typing import Optional, Union
from diffusers.utils import BaseOutput



class NoiseScheduler:
    scheduler: Union[
        DDIMScheduler,
        EulerDiscreteScheduler,
        EulerAncestralDiscreteScheduler,
        DPMSolverMultistepScheduler,
        PNDMScheduler,
        UniPCMultistepScheduler,
    ]
    scheduler_name: str = "euler"

    # //////////////////////////////////////////////////////////////////////////////////// #
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        model_type: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        scheduler_name: Optional[str] = None,
        **kwargs,
    ):     
    # //////////////////////////////////////////////////////////////////////////////////// #
        # Инитится планировщик (по-умолчанию из эйлера)
        scheduler_name = scheduler_name or "euler"
        self.scheduler = EulerDiscreteScheduler.from_pretrained(
            model_path,
            subfolder='scheduler'
        )
        if scheduler_name == "DDIM":
            self.scheduler = DDIMScheduler.from_pretrained(
                model_path,
                subfolder='scheduler'
            )
        elif scheduler_name == "euler":
            self.scheduler = EulerDiscreteScheduler.from_pretrained(
                model_path,
                subfolder='scheduler'
            )
        elif scheduler_name == "euler_a":
            self.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
                model_path,
                subfolder='scheduler'
            )
        elif scheduler_name == "DPM++ 2M":
            self.scheduler = DPMSolverMultistepScheduler.from_pretrained(
                model_path,
                subfolder='scheduler'
            )
        elif scheduler_name == "DPM++ 2M Karras":
            self.scheduler = DPMSolverMultistepScheduler.from_pretrained(
                model_path,
                subfolder='scheduler',
                use_karras_sigmas=True,
            )
        elif scheduler_name == "DPM++ 2M SDE Karras":
            self.scheduler = DPMSolverMultistepScheduler.from_pretrained(
                model_path,
                subfolder='scheduler',
                use_karras_sigmas=True,
                algorithm_type="sde-dpmsolver++",
            )
        elif scheduler_name == "PNDM":
            self.scheduler = PNDMScheduler.from_pretrained(
                model_path,
                subfolder='scheduler'
            )
        elif scheduler_name == "uni_pc":
            self.scheduler = UniPCMultistepScheduler.from_pretrained(
                model_path,
                subfolder='scheduler'
            )
        else:
            raise ValueError(f'Unknown scheduler name: {scheduler_name}')
        
        self.name = scheduler_name
        print(f"Scheduler has successfully changed to '{scheduler_name}'")

        self.dtype = dtype
        self.device = torch.device(device)
        self.path = model_path

    @property
    def scale_factor(self):
        return self.scheduler.init_noise_sigma
    
    @property
    def num_train_timesteps(self):
        return self.scheduler.config.num_train_timesteps
    
    @property
    def order(self):
        return self.scheduler.order
    
    def reload(
        self, 
        scheduler_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        if self.name is not scheduler_name:
            self.__init__(self.path, device, dtype)
    # //////////////////////////////////////////////////////////////////////////////////// #

    
    
    # ==================================================================================== #
    def __call__(
        self, 
        timestep: int, 
        noisy_sample: torch.FloatTensor,
        noise_predict: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:        
    # ==================================================================================== #
        """
        Если передано предсказание шума:
            Вычисляет шумный sample с предыдущего шага 
            noisy_sample[t] -> noisy_sample[t-1] 
        Если предсказание шума не передано:
            То скейлит noisy_sample с текущего шага 
            noisy_sample[t] -> scaled_noisy_sample[t]
        """
        return (
            self.scheduler.step(
                timestep=timestep,
                sample=noisy_sample,
                model_output=noise_predict,
            )
            if noise_predict is not None else
            self.scheduler.scale_model_input(
                sample=noisy_sample,
                timestep=timestep,
            )
        )
    # ==================================================================================== #