from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union, List
import os

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import (
    _create_4d_causal_attention_mask,
    _prepare_4d_attention_mask,
    _prepare_4d_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ModelOutput,
)

from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.models.roberta.configuration_roberta import RobertaConfig

from transformers.models.clip.configuration_clip import (
    CLIPConfig,
    CLIPTextConfig,
    CLIPVisionConfig,
)
from transformers.models.clip.modeling_clip import CLIPModel
from transformers.models.roberta.modeling_roberta import (
    RobertaEmbeddings,
    RobertaEncoder,
    RobertaModel,
)
from transformers.utils import ModelOutput, logging

logger = logging.get_logger(__name__)


@dataclass
class CLIPForSarcasmDetectionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    fuse_embeds: torch.FloatTensor = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            (
                self[k]
                if k not in ["text_model_output", "vision_model_output"]
                else getattr(self, k).to_tuple()
            )
            for k in self.keys()
        )

class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(
            pixel_values.to(dtype=target_dtype)
        )  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class CLIPTextEmbeddings(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, embed_dim
        )

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        seq_length = (
            input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        )

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, is_add_fusion: bool = False):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.is_add_fusion = is_add_fusion

        if is_add_fusion:
            self.cond_hidden_size = config.task_specific_params.get("cond_hidden_size")
            self.is_mlp = config.task_specific_params.get("is_mlp", False)

        if is_add_fusion and self.cond_hidden_size is not None:
            self.gating_param = nn.Parameter(torch.zeros(1))
            self.gated_proj = nn.Linear(self.embed_dim, self.embed_dim)
            if self.is_mlp:
                self.adapter_proj = nn.Sequential(
                    nn.Linear(self.cond_hidden_size, self.embed_dim * 2),
                    nn.ReLU(),
                    nn.Linear(self.embed_dim * 2, self.embed_dim),
                )
            else:
                self.adapter_proj = nn.Linear(self.cond_hidden_size, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        conditional_hidden_states: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        original_tag_len = hidden_states.size(1)

        if conditional_hidden_states is not None:
            hidden_states = (
                torch.cat(
                    [hidden_states, self.adapter_proj(conditional_hidden_states)], dim=1
                )
                if self.is_add_fusion
                else hidden_states
            )

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + causal_attention_mask
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + attention_mask
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            attn_weights_reshaped = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len
            )
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        if not self.is_add_fusion or conditional_hidden_states is None:
            attn_output = self.out_proj(attn_output)
            return attn_output, attn_weights_reshaped

        attn_output = attn_output[:, :original_tag_len, :]
        attn_output = self.out_proj(attn_output) + self.gated_proj(
            attn_output
        ) * F.tanh(self.gating_param)
        return attn_output, attn_weights_reshaped


class CLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer(nn.Module):
    def __init__(self, config: CLIPConfig, is_add_fusion: bool = False):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config, is_add_fusion)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        conditional_hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            conditional_hidden_states=conditional_hidden_states,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class CLIPPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CLIPConfig
    base_model_prefix = "clip"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, CLIPTextEmbeddings):
            module.token_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
            module.position_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
        elif isinstance(module, CLIPVisionEmbeddings):
            factor = self.config.initializer_factor
            nn.init.normal_(
                module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor
            )
            nn.init.normal_(
                module.patch_embedding.weight,
                std=module.config.initializer_range * factor,
            )
            nn.init.normal_(
                module.position_embedding.weight,
                std=module.config.initializer_range * factor,
            )
        elif isinstance(module, CLIPAttention):
            factor = self.config.initializer_factor
            in_proj_std = (
                (module.embed_dim**-0.5)
                * ((2 * module.config.num_hidden_layers) ** -0.5)
                * factor
            )
            out_proj_std = (module.embed_dim**-0.5) * factor
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)

            if module.is_add_fusion:
                nn.init.zeros_(module.gating_param)
                if module.is_mlp:
                    assert isinstance(module.adapter_proj, nn.Sequential)
                    nn.init.normal_(
                        module.adapter_proj[0].weight,
                        std=(module.cond_hidden_size**-0.5) * factor,
                    )
                    nn.init.normal_(
                        module.adapter_proj[2].weight,
                        std=((module.embed_dim * 3) ** -0.5) * factor,
                    )
                else:
                    nn.init.normal_(
                        module.adapter_proj.weight,
                        std=(module.cond_hidden_size**-0.5) * factor,
                    )
                nn.init.normal_(module.gated_proj.weight, std=out_proj_std)

        elif isinstance(module, CLIPMLP):
            factor = self.config.initializer_factor
            in_proj_std = (
                (module.config.hidden_size**-0.5)
                * ((2 * module.config.num_hidden_layers) ** -0.5)
                * factor
            )
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            nn.init.normal_(module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)

        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class CLIPEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`CLIPEncoderLayer`].

    Args:
        config: CLIPConfig
    """

    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config = config

        self.is_conditional = self.config.task_specific_params.get(
            "is_conditional", False
        )
        self.cond_attn_layer_inds = self.config.task_specific_params.get(
            "cond_attn_layer_inds"
        )

        self.layers = nn.ModuleList(
            [
                CLIPEncoderLayer(
                    config,
                    self.is_conditional and idx in self.cond_attn_layer_inds,
                )
                for idx in range(config.num_hidden_layers)
            ]
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: Optional[bool] = False,
        conditional_hidden_states: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        causal_attention_mask = None
        if is_causal:
            # CLIP's text model uses causal mask, prepare it here.
            # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
            causal_attention_mask = _create_4d_causal_attention_mask(
                inputs_embeds.size()[:-1],
                inputs_embeds.dtype,
                device=inputs_embeds.device,
            )

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(
                attention_mask, inputs_embeds.dtype
            )

        concat_causal_attention_mask = None
        if (
            self.is_conditional
            and conditional_hidden_states is not None
            and causal_attention_mask is not None
        ):
            input_shape = inputs_embeds.size()[:-1]
            n = conditional_hidden_states.size(1)
            concat_causal_attention_mask = _create_4d_causal_attention_mask(
                (input_shape[0], input_shape[1] + n),
                inputs_embeds.dtype,
                device=inputs_embeds.device,
            )
        concat_attention_mask = None
        if (
            self.is_conditional
            and conditional_hidden_states is not None
            and attention_mask is not None
        ):
            n = conditional_hidden_states.size(1)
            concat_attention_mask = F.pad(
                attention_mask, (0, n, 0, n), "constant", value=0
            )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            args = (
                (
                    hidden_states,
                    concat_attention_mask,
                    concat_causal_attention_mask,
                    conditional_hidden_states,
                    output_attentions,
                )
                if self.is_conditional and idx in self.cond_attn_layer_inds
                else (
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    None,
                    output_attentions,
                )
            )
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__, *args
                )
            else:
                layer_outputs = encoder_layer(*args)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, encoder_states, all_attentions]
                if v is not None
            )

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )

class CLIPVisionTransformer(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        conditional_hidden_states: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            conditional_hidden_states=conditional_hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class CLIPVisionModel(CLIPPreTrainedModel):
    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        self.vision_model = CLIPVisionTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        conditional_hidden_states: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        return self.vision_model(
            pixel_values=pixel_values,
            conditional_hidden_states=conditional_hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

class HybridConfig(PretrainedConfig):
    model_type = "clip_text_model"
    def __init__(
        self,
        pho_ckpt_name = "vinai/phobert-base",
        clip_ckpt_name = "openai/clip-vit-base-patch32" ,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.pho_num_hidden_layers = kwargs.pop("pho_num_hidden_layers", 6)
        self.clip_num_hidden_layers = kwargs.pop("clip_num_hidden_layers", 6)

        pho_config = kwargs.pop("pho_config", None)
        clip_config = kwargs.pop("clip_config", None)


        if pho_config is None:
            # logger.warning(f"Not found pho_config, load base config from pretrain {pho_ckpt_name}")
            pho_config = RobertaConfig.from_pretrained(pho_ckpt_name)
        else:
            pho_config = RobertaConfig.from_dict(pho_config)

        if clip_config is None:
            # logger.warning(f"Not found clip_config, load base config from preatrain {clip_ckpt_name}")
            clip_config = CLIPTextConfig.from_pretrained(clip_ckpt_name)
            clip_config.task_specific_params = {
                "cond_hidden_size": 768,
                "is_conditional": False,
                "cond_attn_layer_inds": [],
            }
        else:
            clip_config = CLIPTextConfig.from_dict(clip_config)


        pho_config.num_hidden_layers = self.pho_num_hidden_layers
        clip_config.num_hidden_layers = self.clip_num_hidden_layers


        self.pho_config = pho_config.to_dict()
        self.clip_config = clip_config.to_dict()
        self.pho_ckpt_name = pho_ckpt_name
        self.clip_ckpt_name = clip_ckpt_name

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        config_dict["clip_config"]["task_specific_params"] = kwargs.pop("task_specific_params")

        config_dict["pho_config"]["pho_num_hidden_layers"] = kwargs.pop("pho_num_hidden_layers", 6)
        config_dict["clip_config"]["clip_num_hidden_layers"] = kwargs.pop("clip_num_hidden_layers", 6)

        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def from_dict(cls, config_dict, **kwargs) -> "PretrainedConfig":
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        # Those arguments may be passed along for our internal telemetry.
        # We remove them so they don't appear in `return_unused_kwargs`.
        kwargs.pop("_from_auto", None)
        kwargs.pop("_from_pipeline", None)
        # The commit hash might have been updated in the `config_dict`, we don't want the kwargs to erase that update.
        if "_commit_hash" in kwargs and "_commit_hash" in config_dict:
            kwargs["_commit_hash"] = config_dict["_commit_hash"]

        # We remove it from kwargs so that it does not appear in `return_unused_kwargs`.
        config_dict["attn_implementation"] = kwargs.pop("attn_implementation", None)
        config = cls(**config_dict)

        if hasattr(config, "pruned_heads"):
            config.pruned_heads = {int(key): value for key, value in config.pruned_heads.items()}

        # Update config with kwargs if needed
        if "num_labels" in kwargs and "id2label" in kwargs:
            num_labels = kwargs["num_labels"]
            id2label = kwargs["id2label"] if kwargs["id2label"] is not None else []
            if len(id2label) != num_labels:
                raise ValueError(
                    f"You passed along `num_labels={num_labels }` with an incompatible id to label map: "
                    f"{kwargs['id2label']}. Since those arguments are inconsistent with each other, you should remove "
                    "one of them."
                )
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                current_attr = getattr(config, key)
                # To authorize passing a custom subconfig as kwarg in models that have nested configs.
                if isinstance(current_attr, PretrainedConfig) and isinstance(value, dict):
                    value = current_attr.__class__(**value)
                setattr(config, key, value)
                if key != "torch_dtype":
                    to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        if return_unused_kwargs:
            return config, kwargs
        else:
            return config


class HybridEncoder(nn.Module):
    def __init__(self, config: HybridConfig):
        super().__init__()

        self.pho_encoder = RobertaEncoder(RobertaConfig.from_dict(config.pho_config))
        self.linear_map = nn.Sequential(
                    nn.Linear(config.pho_config["hidden_size"], config.clip_config["hidden_size"] * 2),
                    nn.ReLU(),
                    nn.Linear(config.clip_config["hidden_size"] * 2, config.clip_config["hidden_size"]),
                )
        self.clip_encoder = CLIPEncoder(CLIPTextConfig.from_dict(config.clip_config))

    def forward(
        self,

        hidden_states: torch.Tensor,
        pho_attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,

        clip_attention_mask: Optional[torch.FloatTensor] = None,
        is_causal: Optional[bool] = False,
        conditional_hidden_states: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

    ) -> Union[Tuple, BaseModelOutput]:

        roberta_hidden_states = self.pho_encoder(
            hidden_states=hidden_states,
            attention_mask=pho_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        clip_input_hidden_states = self.linear_map(roberta_hidden_states[0])

        clip_outputs = self.clip_encoder(
            inputs_embeds=clip_input_hidden_states,
            attention_mask=clip_attention_mask,
            is_causal=is_causal,
            conditional_hidden_states=conditional_hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return clip_outputs


class HypridPreTrainedModel(PreTrainedModel):
    config_class = HybridConfig
    base_model_prefix = "hybrid"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RobertaEmbeddings", "RobertaSelfAttention", "RobertaSdpaSelfAttention"]

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.clip_config["initializer_factor"]

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.clip_config["initializer_range"])
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.clip_config["initializer_range"])
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, CLIPAttention):
            in_proj_std = (
                (module.embed_dim**-0.5)
                * ((2 * module.config.num_hidden_layers) ** -0.5)
                * factor
            )
            out_proj_std = (module.embed_dim**-0.5) * factor
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)

            if module.is_add_fusion:
                nn.init.zeros_(module.gating_param)
                if module.is_mlp:
                    assert isinstance(module.adapter_proj, nn.Sequential)
                    nn.init.normal_(
                        module.adapter_proj[0].weight,
                        std=(module.cond_hidden_size**-0.5) * factor,
                    )
                    nn.init.normal_(
                        module.adapter_proj[2].weight,
                        std=((module.embed_dim * 3) ** -0.5) * factor,
                    )
                else:
                    nn.init.normal_(
                        module.adapter_proj.weight,
                        std=(module.cond_hidden_size**-0.5) * factor,
                    )
                nn.init.normal_(module.gated_proj.weight, std=out_proj_std)

        elif isinstance(module, CLIPMLP):
            in_proj_std = (
                (module.config.hidden_size**-0.5)
                * ((2 * module.config.num_hidden_layers) ** -0.5)
                * factor
            )
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            nn.init.normal_(module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)

        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

        # elif isinstance(module, RobertaEmbeddings):
        #     module.load_state_dict(self.pho_model.embeddings.state_dict(), strict=False)

        # elif isinstance(module, RobertaEncoder):
        #     module.layer.load_state_dict(self.pho_model.encoder.layer[:self.config.pho_num_hidden_layers].state_dict(), strict=False)

        # elif isinstance(module, CLIPEncoder):
        #     module.layers.load_state_dict(self.clip_model.text_model.encoder.layers[:self.config.clip_num_hidden_layers].state_dict(), strict=False)

        # elif isinstance(module, nn.LayerNorm) and module.normalized_shape[0] == 512:
        #     module.load_state_dict(self.clip_model.text_model.final_layer_norm.state_dict(), strict=False)

        # elif isinstance(module, CLIPAttention):
        #     out_proj_std = (module.embed_dim**-0.5) * factor

        #     if module.is_add_fusion:
        #         nn.init.zeros_(module.gating_param)
        #         if module.is_mlp:
        #             assert isinstance(module.adapter_proj, nn.Sequential)
        #             nn.init.normal_(
        #                 module.adapter_proj[0].weight,
        #                 std=(module.cond_hidden_size**-0.5) * factor,
        #             )
        #             nn.init.normal_(
        #                 module.adapter_proj[2].weight,
        #                 std=((module.embed_dim * 3) ** -0.5) * factor,
        #             )
        #         else:
        #             nn.init.normal_(
        #                 module.adapter_proj.weight,
        #                 std=(module.cond_hidden_size**-0.5) * factor,
        #             )
        #         nn.init.normal_(module.gated_proj.weight, std=out_proj_std)

        # if isinstance(module, nn.Linear) and module.bias is not None:
        #     module.bias.data.zero_()

    def post_init(self):
        self.pho_model = RobertaModel.from_pretrained(self.config.pho_ckpt_name)
        self.clip_model = CLIPModel.from_pretrained(self.config.clip_ckpt_name)

        self.init_weights()
        self._backward_compatibility_gradient_checkpointing()

        del self.pho_model, self.clip_model
        self.pho_model = None
        self.clip_model = None

class CLIPTextTransformer(nn.Module):

    def __init__(self, config: HybridConfig, warn_if_padding_and_no_attention_mask, get_extended_attention_mask, invert_attention_mask, get_head_mask):
        super().__init__()
        self.clip_config = CLIPTextConfig.from_dict(config.clip_config)
        self.pho_config = RobertaConfig.from_dict(config.pho_config)
        self.embeddings = RobertaEmbeddings(self.pho_config)
        self.encoder = HybridEncoder(config)
        self.final_layer_norm = nn.LayerNorm(self.clip_config.hidden_size, eps=self.clip_config.layer_norm_eps)

        self.attn_implementation = self.pho_config._attn_implementation
        self.position_embedding_type = self.pho_config.position_embedding_type

        # For `pooled_output` computation
        self.eos_token_id = self.clip_config.eos_token_id

        self.warn_if_padding_and_no_attention_mask = warn_if_padding_and_no_attention_mask
        self.get_extended_attention_mask = get_extended_attention_mask
        self.invert_attention_mask = invert_attention_mask
        self.get_head_mask = get_head_mask



    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,

        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

        conditional_hidden_states: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:

        output_attentions = output_attentions if output_attentions is not None else self.pho_config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.pho_config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.pho_config.use_return_dict

        if self.pho_config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.pho_config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask) #Pretrained
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=device)

        use_sdpa_attention_masks = (
            self.attn_implementation == "sdpa"
            and self.position_embedding_type == "absolute"
            and head_mask is None
            and not output_attentions
        )

        if use_sdpa_attention_masks and attention_mask.dim() == 2:
            if self.pho_config.is_decoder:
                extended_attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    input_shape,
                    embedding_output,
                    past_key_values_length,
                )
            else:
                extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    attention_mask, embedding_output.dtype, tgt_len=seq_length
                )
        else:
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape) #PretrainedModel

        if self.pho_config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if use_sdpa_attention_masks and encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    encoder_attention_mask, embedding_output.dtype, tgt_len=seq_length
                )
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask) #PretrainedModel

        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.pho_config.num_hidden_layers) #PretrainedModel

        # ------------------------------------------------------------------------------- #

        encoder_outputs = self.encoder(
            embedding_output,
            pho_attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,

            clip_attention_mask=attention_mask,
            is_causal=True,
            conditional_hidden_states=conditional_hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # ------------------------------------------------------------------------------- #

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        if self.eos_token_id == 2:
            pooled_output = last_hidden_state[
                torch.arange(
                    last_hidden_state.shape[0], device=last_hidden_state.device
                ),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(
                    dim=-1
                ),
            ]
        else:
            pooled_output = last_hidden_state[
                torch.arange(
                    last_hidden_state.shape[0], device=last_hidden_state.device
                ),
                # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
                (
                    input_ids.to(dtype=torch.int, device=last_hidden_state.device)
                    == self.eos_token_id
                )
                .int()
                .argmax(dim=-1),
            ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class CLIPTextModel(HypridPreTrainedModel):
    config_class = HybridConfig

    _no_split_modules = ["RobertaEmbeddings", "CLIPEncoderLayer", "RobertaLayer"]

    def __init__(self, config: HybridConfig):
        super().__init__(config)
        self.text_model = CLIPTextTransformer(
            config,
            self.warn_if_padding_and_no_attention_mask,
            self.get_extended_attention_mask,
            self.invert_attention_mask,
            self.get_head_mask
        )
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        conditional_hidden_states: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            conditional_hidden_states=conditional_hidden_states,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )