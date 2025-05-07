from typing import Any, Dict, Optional
import torch
from transformers import GenerationMixin, GenerationConfig


class NovaGenerationMixin(GenerationMixin):
    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
            if 'nova_attention_mask' in model_kwargs:
                bsz, L = model_kwargs['nova_attention_mask'].size()[:2]

                model_kwargs['no_mask_idx'] = torch.cat([
                    model_kwargs['no_mask_idx'], torch.zeros((bsz, 1)).fill_(L).type_as(model_kwargs['no_mask_idx'])
                ], dim=-1)

                nova_attention_mask = torch.zeros((bsz, L + 1, L + 1)).type_as(model_kwargs['nova_attention_mask'])
                nova_attention_mask[:, :L, :L] = model_kwargs['nova_attention_mask']
                for idx in range(bsz):
                    nova_attention_mask[idx, -1, model_kwargs['no_mask_idx'][idx]] = 1
                model_kwargs['nova_attention_mask'] = nova_attention_mask
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )

        if "cache_position" in model_kwargs and model_kwargs["cache_position"] is not None:
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + 1

        return model_kwargs

    def _reorder_cache(self, past_key_values, beam_idx):
        raise NotImplementedError(
            f"Make sure that a `_reorder_cache` function is correctly implemented in {self.__class__.__module__} to"
            f" enable beam search for {self.__class__}"
        )
