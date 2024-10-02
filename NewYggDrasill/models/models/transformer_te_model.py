import torch

from transformers import (
    T5EncoderModel,
    T5TokenizerFast
)
from typing import List, Optional, Union
from diffusers.utils.peft_utils import scale_lora_layers, unscale_lora_layers



class TransformerTextEncoderModel:
    tokenizer: T5TokenizerFast
    transformer: T5EncoderModel 

     # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        **kwargs,
    ):  
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
        self.tokenizer = T5TokenizerFast.from_pretrained(model_path)
        self.transformer = T5EncoderModel.from_pretrained(
            model_path,
            # 
        )

        # Инициализируем константы
        self.dtype = dtype
        self.model_path = model_path
        self.device = torch.device(device)

        print(f"TransformerTextEncoderModel model has successfully loaded from '{model_path}' checkpoint!")
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #


    
    # ================================================================================================================ #
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs
    ):
    # ================================================================================================================ #
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        # untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

        # if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        #     removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
        #     # logger.warning(
        #     #     "The following part of your input was truncated because `max_sequence_length` is set to "
        #     #     f" {max_sequence_length} tokens: {removed_text}"
        #     # )

        prompt_embeds = self.text_encoder_2(
            text_input_ids.to(device), 
            output_hidden_states=False
        )[0]

        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=self.device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds
    # ================================================================================================================ #





























# # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._get_t5_prompt_embeds
# def _get_t5_prompt_embeds(
#     self,
#     prompt: Union[str, List[str]] = None,
#     num_images_per_prompt: int = 1,
#     max_sequence_length: int = 512,
#     device: Optional[torch.device] = None,
#     dtype: Optional[torch.dtype] = None,
# ):
#     device = device or self._execution_device
#     dtype = dtype or self.text_encoder.dtype

#     prompt = [prompt] if isinstance(prompt, str) else prompt
#     batch_size = len(prompt)

#     text_inputs = self.tokenizer_2(
#         prompt,
#         padding="max_length",
#         max_length=max_sequence_length,
#         truncation=True,
#         return_length=False,
#         return_overflowing_tokens=False,
#         return_tensors="pt",
#     )
#     text_input_ids = text_inputs.input_ids
#     untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

#     if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
#         removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
#         logger.warning(
#             "The following part of your input was truncated because `max_sequence_length` is set to "
#             f" {max_sequence_length} tokens: {removed_text}"
#         )

#     prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]

#     dtype = self.text_encoder_2.dtype
#     prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

#     _, seq_len, _ = prompt_embeds.shape

#     # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
#     prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
#     prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

#     return prompt_embeds