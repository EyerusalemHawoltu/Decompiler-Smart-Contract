import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
from transformers import LlamaModel, LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LLAMA_ATTENTION_CLASSES, LlamaMLP, LlamaRMSNorm
from transformers.models.llama.modeling_llama import LlamaSdpaAttention, apply_rotary_pos_emb, repeat_kv
from transformers import logging, Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from generation_utils import NovaGenerationMixin

logger = logging.get_logger(__name__)


class NovaTokenizer():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.labels = set([
            tokenizer.encode(f'<label-{i}>')[-1] for i in range(1, 257)
        ])

    def encode(self, input_text: str, output_text: str, char_types: str):
        assert len(input_text + output_text) > 0, "`input_text` + `output_text` should not be empty."
        assert len(input_text + output_text) == len(char_types), "`char_types` should be a string of `01` with the same length of `input_text` + `output_text`."

        # input
        input_text_lst = []
        start = 0
        for i in range(1, len(input_text)):
            if char_types[i] != char_types[i - 1]:
                input_text_lst.append([input_text[start: i], char_types[i - 1]])
                start = i
        if input_text != '':
            input_text_lst.append([input_text[start: ], char_types[: len(input_text)][-1]])
        
        # output
        output_text_lst = []
        start = 0
        for i in range(1, len(output_text)):
            if char_types[len(input_text) + i] != char_types[len(input_text) + i - 1]:
                output_text_lst.append([output_text[start: i], char_types[len(input_text) + i - 1]])
                start = i
        if output_text != '':
            output_text_lst.append([output_text[start: ], char_types[-1]])

        input_ids = []
        output_ids = []
        tokenized_text_lst = []
        l = 0
        for txt, ty in input_text_lst:
            # remove bos from Llama's tokenization
            txt_ids = self.tokenizer.encode(txt)
            if txt_ids[0] == self.tokenizer.bos_token_id:
                txt_ids = txt_ids[1: ]
            tokenized_text_lst.append([txt_ids, ty])
            
            input_ids += txt_ids
            output_ids += [-100] * len(txt_ids)
            l += len(txt_ids)
        for txt, ty in output_text_lst:
            # remove bos from Llama's tokenization
            txt_ids = self.tokenizer.encode(txt)
            if txt_ids[0] == self.tokenizer.bos_token_id:
                txt_ids = txt_ids[1: ]
            tokenized_text_lst.append([txt_ids, ty])
            
            input_ids += txt_ids
            output_ids += txt_ids
            l += len(txt_ids)
        
        input_ids = np.array(input_ids, dtype=np.int32)
        output_ids = np.array(output_ids, dtype=np.int32)
        attention_mask = np.zeros((l, l))
        cur_len = 0
        no_mask_idx = []
        for text_ids, ty in tokenized_text_lst:
            input_ids[cur_len: cur_len + len(text_ids)] = text_ids
            
            if ty == "1":
                sub_text_ids_lst = []
                start = 0
                for i, e in enumerate(text_ids):
                    if e in self.labels and i + 1 < len(text_ids) and text_ids[i + 1] == self.tokenizer.encode('\n')[-1]:
                        sub_text_ids_lst.append(text_ids[start: i + 1])
                        start = i + 1
                if start < len(text_ids):
                    sub_text_ids_lst.append(text_ids[start: ])
                sub_cur_len = 0
                for sub_text_ids in sub_text_ids_lst:
                    f = np.ones((len(sub_text_ids), len(sub_text_ids)))
                    # f.fill(0.9)
                    attention_mask[cur_len + sub_cur_len: cur_len + sub_cur_len + len(sub_text_ids), 
                                    cur_len + sub_cur_len: cur_len + sub_cur_len + len(sub_text_ids)] = \
                                        np.tril(f)
                    
                    if cur_len + sub_cur_len - 1 >= 0:
                        attention_mask[cur_len + sub_cur_len: cur_len + sub_cur_len + len(sub_text_ids), cur_len + sub_cur_len - 1] = 1
                    if len(no_mask_idx) > 0:
                        attention_mask[cur_len + sub_cur_len + len(sub_text_ids) - 1, np.array(no_mask_idx)] = 1
                    
                    no_mask_idx += [cur_len + sub_cur_len + len(sub_text_ids) - 1]
                    sub_cur_len += len(sub_text_ids)

            elif ty == "0":
                attention_mask[cur_len: cur_len + len(text_ids), cur_len: cur_len + len(text_ids)] = np.tril(
                    np.ones(
                        (len(text_ids), len(text_ids))
                    )
                )
                if len(no_mask_idx) > 0:
                    attention_mask[
                        cur_len: cur_len + len(text_ids), np.array(no_mask_idx)
                    ] = 1
                no_mask_idx += [idx for idx in range(cur_len, cur_len + len(text_ids))]
                
            cur_len += len(text_ids)

        return {
            'input_ids': input_ids, 'labels': output_ids, 'nova_attention_mask': attention_mask.astype(bool), 
            'no_mask_idx': no_mask_idx
        }


class NovaAttention(LlamaSdpaAttention):
    def forward_output_attentions(
            self, 
            hidden_states,
            attention_mask,
            nova_attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
        ):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        query_states_1, query_states_2 = torch.split(query_states, self.num_heads // 2, dim=1)
        key_states_1, key_states_2 = torch.split(key_states, self.num_heads // 2, dim=1)
        value_states_1, value_states_2 = torch.split(value_states, self.num_heads // 2, dim=1)

        attn_weights_1 = torch.matmul(query_states_1, key_states_1.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights_2 = torch.matmul(query_states_2, key_states_2.transpose(2, 3)) / math.sqrt(self.head_dim)

        # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights_1 = attn_weights_1 + causal_mask
        attn_weights_2 = attn_weights_2 + nova_attention_mask
        
        attn_weights_1 = nn.functional.softmax(attn_weights_1, dim=-1, dtype=torch.float32).to(query_states_1.dtype)
        attn_weights_1 = nn.functional.dropout(attn_weights_1, p=self.attention_dropout, training=self.training)
        attn_output_1 = torch.matmul(attn_weights_1, value_states_1)
        attn_weights_2 = nn.functional.softmax(attn_weights_2, dim=-1, dtype=torch.float32).to(query_states_2.dtype)
        attn_weights_2 = nn.functional.dropout(attn_weights_2, p=self.attention_dropout, training=self.training)
        attn_output_2 = torch.matmul(attn_weights_2, value_states_2)

        attn_weights = torch.cat([attn_weights_1, attn_weights_2], dim=1)
        attn_output = torch.cat([attn_output_1, attn_output_2], dim=1)

        # upcast attention to fp32
        # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        # attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        nova_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            return self.forward_output_attentions(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                nova_attention_mask=nova_attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2) # [B, num, L, h]
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)   # [B, ?, L, h]
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)   # [B, ?, L, h]

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # In case static cache is used, it is an instance attribute.
        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)       # [B, num, L, h]
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        
        # Nova split attention
        # nova_h = self.config.nova_num_heads
        # query_states_1, query_states_2 = query_states[:, :-nova_h, :, :], query_states[:, -nova_h:, :, :]
        # key_states_1, key_states_2 = key_states[:, :-nova_h, :, :], key_states[:, -nova_h:, :, :]
        # value_states_1, value_states_2 = value_states[:, :-nova_h, :, :], value_states[:, -nova_h:, :, :]
        query_states_1, query_states_2 = torch.split(query_states, self.num_heads // 2, dim=1)
        key_states_1, key_states_2 = torch.split(key_states, self.num_heads // 2, dim=1)
        value_states_1, value_states_2 = torch.split(value_states, self.num_heads // 2, dim=1)

        # standard attention
        attn_output_1 = torch.nn.functional.scaled_dot_product_attention(
            query_states_1,
            key_states_1,
            value_states_1,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=causal_mask is None and q_len > 1,
        )
        
        # Nova attention
        attn_output_2 = torch.nn.functional.scaled_dot_product_attention(
            query_states_2,
            key_states_2,
            value_states_2,
            attn_mask=nova_attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        
        attn_output = torch.cat([attn_output_1, attn_output_2], dim=1)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class NovaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size

        self.self_attn = NovaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        nova_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            nova_attention_mask=nova_attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class NovaModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [NovaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        nova_attention_mask: Optional[torch.Tensor] = None,
        no_mask_idx: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError("cache_position is a required argument when using StaticCache.")
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_seen_tokens)

        # apply the nova attention
        if nova_attention_mask is not None:
            bsz, L = inputs_embeds.size()[:2]
            nova_attention_mask = nova_attention_mask.unsqueeze(1).type(inputs_embeds.dtype)
            # nova_attention_mask = (nova_attention_mask - 1) * torch.finfo(inputs_embeds.dtype).max
            nova_attention_mask = (nova_attention_mask - 1) * 1.e32
            nova_attention_mask = nova_attention_mask[:, :, -L:, :]

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    nova_attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    nova_attention_mask=nova_attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class NovaForCausalLM(LlamaForCausalLM, NovaGenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.model = NovaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        nova_attention_mask: Optional[torch.Tensor] = None,
        no_mask_idx: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            nova_attention_mask=nova_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs
    ):
        # With static cache, the `past_key_values` is None
        # TODO joao: standardize interface for the different Cache classes and remove of this if
        # print('prepare input:', input_ids.size(), kwargs.get("nova_attention_mask").size(), kwargs.get("no_mask_idx").size())
        
        has_static_cache = False
        if past_key_values is None:
            past_key_values = getattr(getattr(self.model.layers[0], "self_attn", {}), "past_key_value", None)
            has_static_cache = past_key_values is not None

        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                max_cache_length = (
                    torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                    if past_key_values.get_max_length() is not None
                    else None
                )
                cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)
            # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        else:
            cache_position = cache_position[-input_length:]

        if has_static_cache:
            past_key_values = None

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "nova_attention_mask": kwargs.get("nova_attention_mask"),
                "no_mask_idx": kwargs.get("no_mask_idx")
            }
        )
        return model_inputs
