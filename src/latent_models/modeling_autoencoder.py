from transformers import T5ForConditionalGeneration, T5Config
from transformers.models.t5.modeling_t5 import T5Attention, T5LayerSelfAttention, T5Block, T5LayerFF, T5Stack
import torch.nn as nn
import torch
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.utils import ModelOutput
from typing import Optional
import torch.nn.functional as F
from dataclasses import dataclass
from torch.nn import Module
import math


class CompressionNetAE(nn.Module):
    def __init__(self, in_dim=768, latent_dim=64):
        super().__init__()
        self.norm_in = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, in_dim // 2)  # e.g., 768 → 384 or 512 → 256
        self.norm1 = nn.LayerNorm(in_dim // 2)

        if in_dim == 768:
            self.use_third_layer = True
            self.fc2 = nn.Linear(in_dim // 2, in_dim // 4)  # 384 → 192
            self.norm2 = nn.LayerNorm(in_dim // 4)
            latent_in_dim = in_dim // 4
        elif in_dim == 512:
            self.use_third_layer = False
            latent_in_dim = in_dim // 2
        else:
            raise ValueError("Unsupported in_dim: only 512 or 768 are supported.")

        self.fc_latent = nn.Linear(latent_in_dim, latent_dim)  # → 64

    def forward(self, x):
        x = self.norm_in(x)
        x = F.gelu(self.norm1(self.fc1(x)))

        if self.use_third_layer:
            x = F.gelu(self.norm2(self.fc2(x)))

        z = self.fc_latent(x)
        # z = F.normalize(z, dim=-1) * math.sqrt(z.shape[-1])
        return z


class ReconstructionNet(nn.Module):
    def __init__(self, in_dim=64, out_dim=768):
        super().__init__()
        if out_dim == 768:
            self.use_third_layer = True
            mid1 = out_dim // 4  # 192
            mid2 = out_dim // 2  # 384
        elif out_dim == 512:
            self.use_third_layer = False
            mid1 = out_dim // 2  # 256
        else:
            raise ValueError("Unsupported out_dim: only 512 or 768 are supported.")

        self.norm_in = nn.LayerNorm(in_dim)
        self.linear1 = nn.Linear(in_dim, mid1)
        self.norm1 = nn.LayerNorm(mid1)

        if self.use_third_layer:
            self.linear2 = nn.Linear(mid1, mid2)
            self.norm2 = nn.LayerNorm(mid2)
            self.linear3 = nn.Linear(mid2, out_dim)
        else:
            self.linear2 = nn.Linear(mid1, out_dim)

    def forward(self, x):
        x = self.norm_in(x)
        x = F.gelu(self.norm1(self.linear1(x)))

        if self.use_third_layer:
            x = F.gelu(self.norm2(self.linear2(x)))
            x = self.linear3(x)
        else:
            x = self.linear2(x)

        return x


class AutoencoderEncoderAttention(T5Attention):
    def __init__(self, config, dtype, mem_len, num_heads):
        super().__init__(config, has_relative_attention_bias=True)
        self.is_decoder = False
        self.dtype = dtype
        self.mem_len = mem_len
        self.num_heads = num_heads

        # learnable bias for mem↔mem attention
        # self.mem_position_bias = nn.Parameter(torch.zeros(self.num_heads, mem_len, mem_len))  # [h, mem_len, mem_len]

        # chunk-aligned relative bias embedding (e.g., max 32 chunks)
        self.max_relative_chunk = 32
        self.chunk_relative_bias = nn.Embedding(self.max_relative_chunk, self.num_heads)

    def compute_chunk_relative_bias(self, mem_len, text_len, device):
        """
        Generate mem->text chunk-based bias, with safe handling when mem_len > text_len.
        """
        # 如果任一维度为 0，直接返回空 tensor
        if mem_len <= 0 or text_len <= 0:
            return torch.zeros(self.num_heads, mem_len, text_len, device=device, dtype=self.dtype)

        # 1. 用 ceil(text_len / mem_len) 来计算 chunk_size，保证至少为 1
        #    这样即使 text_len < mem_len，也不会出现 chunk_size == 0
        chunk_size = (text_len + mem_len - 1) // mem_len

        # 2. 初始化输出 [num_heads, mem_len, text_len]
        rel_bias = torch.zeros(self.num_heads, mem_len, text_len, device=device, dtype=self.dtype)

        # 3. 填充偏置
        for i in range(mem_len):
            for j in range(text_len):
                # j // chunk_size 不会再除以 0
                rel_idx = abs(i - (j // chunk_size))
                rel_idx = min(rel_idx, self.max_relative_chunk - 1)
                # 这里将整型 rel_idx 打包成 tensor，再通过 embedding lookup
                rel_bias[:, i, j] = self.chunk_relative_bias(torch.tensor(rel_idx, device=device))

        return rel_bias

    def forward(self, hidden_states, mask,
                position_bias=None, output_attentions=False):
        # hidden_states -> b,L+L_dfg+L_ast,d
        # mask -> b,L+L_dfg+L_ast -> 0 or 1
        # position_bias -> bsz, h, L', L' -> code-code and leaf-leaf positional attention and all masks

        bsz = hidden_states.size()[0]
        position_bias_is_none = position_bias is None

        # get queries
        query_states = self.q(hidden_states)  # b,L',h*dk
        query_states = query_states.view(bsz, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)  # b,h,L',dk

        # get keys
        key_states = self.k(hidden_states)  # b,L',h*dk
        key_states = key_states.view(bsz, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)  # b,h,L',dk

        # get values
        value_states = self.v(hidden_states)  # b,L',h*dk
        value_states = value_states.view(bsz, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)  # b,h,L',dk

        # content attention
        scores = torch.matmul(query_states, key_states.transpose(3, 2))  # b,h,L',L'

        if position_bias_is_none:
            device = hidden_states.device
            L = hidden_states.size(1)
            mem_len = self.mem_len
            text_len = L - mem_len

            # Base T5 relative bias: only for text↔text
            base_bias = self.compute_bias(text_len, text_len, device=device)  # [h, T, T]

            # Init full bias tensor
            position_bias = torch.zeros(self.n_heads, L, L, device=device)

            # Fill text↔text (bottom-right)
            position_bias[:, mem_len:, mem_len:] = base_bias

            # Fill mem↔mem (top-left): use learnable bias
            # position_bias[:, :mem_len, :mem_len] = self.mem_position_bias
            position_bias[:, :mem_len, :mem_len] = self.compute_bias(mem_len, mem_len, device=device)

            # Fill mem→text (top-right): use chunk-aligned
            chunk_bias = self.compute_chunk_relative_bias(mem_len, text_len, device=device)  # [h, mem, text]
            position_bias[:, :mem_len, mem_len:] = chunk_bias

            # Fill text→mem (bottom-left): set to -inf
            position_bias[:, mem_len:, :mem_len] = float("-inf")

            # Broadcast to batch
            position_bias = position_bias.unsqueeze(0).expand(bsz, -1, -1, -1)  # [bsz, h, L, L]

            # Optional: apply mask (e.g., attention mask)
            if mask is not None:
                neg_inf = torch.finfo(self.dtype).min
                pad_mask = 1 - mask  # (b, L)
                mask_pad = (pad_mask[:, None, :, None]  # (b,1,L,1)
                            + pad_mask[:, None, None, :])  # (b,1,1,L) → (b,1,L,L)
                mask_pad = mask_pad.to(dtype=self.dtype)
                mask_pad = torch.clamp(mask_pad, max=1) * neg_inf

                position_bias = position_bias + mask_pad

        scores += position_bias

        # softmax and dropout
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)  # b,h,L',L'
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)  # b,h,L',L'

        # multiply attentions with values and add
        attn_output = torch.matmul(attn_weights, value_states)  # b,h,L',dk
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, -1, self.inner_dim)  # b,L',h*dk
        attn_output = self.o(attn_output)  # b,L',d

        # outputs
        outputs = (attn_output,
                   position_bias if position_bias_is_none else None,
                   attn_weights if output_attentions else None)
        return outputs


class AutoencoderEncoderLayerSelfAttention(T5LayerSelfAttention):
    def __init__(self, config, dtype, mem_len, num_heads):
        super().__init__(config, has_relative_attention_bias=True)
        self.SelfAttention = AutoencoderEncoderAttention(config, dtype, mem_len, num_heads)

    def forward(self, hidden_states, mask,
                position_bias=None, output_attentions=False):
        # hidden_states -> b,L+L_dfg+L_ast,d
        # mask -> b,L+L_dfg+L_ast -> 0 or 1
        # position_bias -> bsz, h, L', L' -> code-code and leaf-leaf positional attention and all masks

        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(normed_hidden_states, mask, position_bias, output_attentions)
        hidden_states = hidden_states + self.dropout(attention_output[0])
        return (hidden_states,) + attention_output[1:]  # (bL'd) (bhL'L') (bhL'L')


class AutoencoderEncoderBlock(T5Block):
    def __init__(self, config, dtype, mem_len, num_heads):
        super().__init__(config, True)
        self.layer = nn.ModuleList()
        self.layer.append(AutoencoderEncoderLayerSelfAttention(config, dtype, mem_len, num_heads))
        self.layer.append(T5LayerFF(config))

    def forward(self, hidden_states, mask, position_bias=None, output_attentions=False):
        # hidden_states -> b,L+L_dfg+L_ast,d
        # mask -> b,L+L_dfg+L_ast -> 0 or 1
        # dfg_code_links -> bsz, L_dfg, L -> 0 or 1
        # dfg_dfg_links -> bsz, L_dfg, L_dfg -> 0 or 1
        # ast_code_links -> bsz, L_ast, L -> 0 or 1
        # ast_ast_sims -> bsz, L_ast, L_ast
        # position_bias -> bsz, h, L', L' -> code-code and leaf-leaf positional attention and all masks

        self_attention_outputs = self.layer[0](hidden_states, mask, position_bias=position_bias,
                                               output_attentions=output_attentions)
        hidden_states = self_attention_outputs[0]

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return (hidden_states,) + self_attention_outputs[1:]  # (bL'd) (bhL'L') (bhL'L')


class AutoencoderEncoderStack(T5Stack):
    def __init__(self, config, mem_len, num_heads):
        super().__init__(config)
        self.block = nn.ModuleList(
            [AutoencoderEncoderBlock(config, self.dtype, mem_len, num_heads) for i in range(config.num_layers)])

    def forward(self, inputs_embeds, attention_mask, output_attentions=False, output_hidden_states=False):
        # inputs_embeds -> b,L+L_dfg+L_ast,d
        # attention_mask -> b,L+L_dfg+L_ast -> 0 or 1
        # dfg_code_links -> bsz, L_dfg, L -> 0 or 1
        # dfg_dfg_links -> bsz, L_dfg, L_dfg -> 0 or 1
        # ast_code_links -> bsz, L_ast, L -> 0 or 1
        # ast_ast_sims -> bsz, L_ast, L_ast

        all_attentions = () if output_attentions else None
        position_bias = None
        hidden_states = self.dropout(inputs_embeds)  # bL'd
        all_hidden_states = (hidden_states,) if output_hidden_states else None

        for i, layer_module in enumerate(self.block):

            layer_outputs = layer_module(hidden_states, attention_mask, position_bias=position_bias,
                                         output_attentions=output_attentions)
            hidden_states = layer_outputs[0]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)

            # share the position biases between the layers - after the first layer store them
            if i == 0:
                position_bias = layer_outputs[1]

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


@dataclass
class AutoencoderOutput(ModelOutput):
    lm_loss: Optional[torch.FloatTensor] = None
    lm_logits: torch.FloatTensor = None
    mmd_loss: Optional[torch.FloatTensor] = None


class AutoencoderForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, args):
        # config = T5Config.from_pretrained('Salesforce/codet5-base')
        config = T5Config.from_pretrained('Salesforce/codet5-small')

        # if args.model_size == 'small':
        #    config.d_model, config.d_kv, config.d_ff = 256, 32, 1024
        #    config.num_layers, config.num_decoder_layers, config.num_heads = 5, 5, 8
        super().__init__(config)
        factor = config.initializer_factor

        self.mem_size = args.mem_slots_len
        self.mem_dim = args.mem_dim

        # memory encoder weights
        self.memory_emb = nn.Parameter(torch.empty((1, 1, self.config.d_model)), requires_grad=True)
        self.memory_emb.data.normal_(mean=0.0, std=factor * 1.0)

        # encoder
        self.encoder = AutoencoderEncoderStack(config, self.mem_size, config.num_heads)

        # self.enc_output_proj = nn.Linear(config.d_model, self.mem_dim)

        self.enc_output_proj = CompressionNetAE(self.config.d_model, self.mem_dim)

        # self.dec_input_proj = nn.Linear(in_features=self.mem_dim, out_features=self.config.d_model)

        self.dec_input_proj = ReconstructionNet(self.mem_dim, self.config.d_model)

        # load pretrained weights
        if args.model_size != 'small':
            print('\nLoading StructCoder weights from CodeT5.')
            pt_model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')
            pt_model_dict = pt_model.state_dict()
            model_dict = self.state_dict()
            not_found = []
            for k in model_dict:
                if k in pt_model_dict:
                    model_dict[k] = pt_model_dict[k]
                else:
                    not_found.append(k)
            self.load_state_dict(model_dict)
            print('Could not load weights "' + str(not_found) + '" from pretrained CodeT5.')

            # Freeze the loaded parameters
            for name, param in self.named_parameters():
                if name in pt_model_dict:  # Only freeze parameters that were loaded from CodeT5
                    param.requires_grad = False

        else:
            print('\nLoading StructCoder weights from CodeT5-small.')
            pt_model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small')
            pt_model_dict = pt_model.state_dict()
            model_dict = self.state_dict()
            not_found = []
            for k in model_dict:
                if k in pt_model_dict:
                    model_dict[k] = pt_model_dict[k]
                else:
                    not_found.append(k)
            self.load_state_dict(model_dict)
            print('Could not load weights "' + str(not_found) + '" from pretrained CodeT5-small.')

            # Freeze the loaded parameters
            for name, param in self.named_parameters():
                if name in pt_model_dict:  # Only freeze parameters that were loaded from CodeT5
                    param.requires_grad = False

        """"""

    # to support generate()
    def forward2(self, encoder_outputs, attention_mask, decoder_input_ids, past_key_values):
        decoder_outputs = self.decoder(input_ids=decoder_input_ids,
                                       encoder_hidden_states=encoder_outputs.last_hidden_state,
                                       encoder_attention_mask=attention_mask,
                                       past_key_values=past_key_values,
                                       use_cache=self.config.use_cache
                                       )
        sequence_output = decoder_outputs[0]  # b,L_out,d
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)  # b,L_out,V
        return Seq2SeqLMOutput(logits=lm_logits,
                               past_key_values=decoder_outputs.past_key_values,
                               decoder_hidden_states=decoder_outputs.hidden_states)

    def compute_mmd(self, z, prior_samples, kernel='imq', sigma=1.0, imq_constant=1.0):
        """
        z: [B, L, D], prior_samples: [B, L, D]
        Flatten along batch and sequence → [B*L, D]

        Args:
            kernel: 'rbf' or 'imq' (Inverse Multiquadratic)
            sigma: bandwidth parameter for RBF kernel
            imq_constant: C parameter for IMQ kernel (k(x,y) = C / (C + ||x-y||^2))
        """
        B, L, D = z.shape
        z = z.view(B * L, D)
        prior_samples = prior_samples.view(B * L, D)

        # Compute pairwise squared distances
        d_zz = torch.cdist(z, prior_samples, p=2).pow(2)
        d_xx = torch.cdist(z, z, p=2).pow(2)
        d_yy = torch.cdist(prior_samples, prior_samples, p=2).pow(2)

        if kernel == 'rbf':
            sigmas = [sigma * s for s in [1.0, 2.0, 4.0]]  # Multi-scale RBF
            k_xx = self._rbf_kernel(d_xx, sigmas).mean()
            k_yy = self._rbf_kernel(d_yy, sigmas).mean()
            k_zz = self._rbf_kernel(d_zz, sigmas).mean()
        elif kernel == 'imq':
            k_xx = self._imq_kernel(d_xx, imq_constant).mean()
            k_yy = self._imq_kernel(d_yy, imq_constant).mean()
            k_zz = self._imq_kernel(d_zz, imq_constant).mean()
        else:
            raise NotImplementedError("Only 'rbf' and 'imq' kernels are implemented")

        mmd = k_xx + k_yy - 2 * k_zz
        return mmd

    def _rbf_kernel(self, d, sigmas):
        """RBF kernel: exp(-d/(2*sigma^2)) averaged over multiple scales"""
        result = 0
        for sigma in sigmas:
            result += torch.exp(-d / (2 * sigma ** 2))
        return result / len(sigmas)

    def _imq_kernel(self, d, C):
        """Inverse Multiquadratic kernel: C / (C + ||x-y||^2)"""
        return C / (C + d)

    # TODO: Add support for output_attentions, output_hidden_states
    def forward(self, inputs=None, outputs=None,
                encoder_outputs=None, attention_mask=None, decoder_input_ids=None,
                past_key_values=None, max_length=None, num_beams=None, mmd_weight=0, **kwargs):
        # inputs['input_ids'] -> bsz, L
        # inputs['attention_mask'] -> bsz, L

        # remaining inputs are for forward2
        if encoder_outputs is not None:
            return self.forward2(encoder_outputs, attention_mask, decoder_input_ids, past_key_values)

        # 1. Input embedding
        device = inputs['input_ids'].device
        origin_input_embeds = self.shared(inputs['input_ids'])  # b,L,d
        origin_attention_mask = inputs['attention_mask']
        L_dfg, L_ast = 0, 0
        bsz, L = inputs['input_ids'].size()

        input_embeds = self.memory_emb * torch.ones((bsz, self.mem_size, 1), device=device)
        attention_mask = torch.ones((bsz, self.mem_size), device=device).int()

        input_embeds = torch.cat((input_embeds, origin_input_embeds), dim=1)  # b,L+L_mem,d
        attention_mask = torch.cat((attention_mask, origin_attention_mask), dim=-1)

        # 2. encoder
        encoder_outputs = self.encoder(input_embeds, attention_mask)

        mem_last_hidden_state = encoder_outputs.last_hidden_state[:, :self.mem_size, :]
        attention_mask = attention_mask[:, :self.mem_size]

        mem_latent = self.enc_output_proj(mem_last_hidden_state)

        dec_input_hidden_state = self.dec_input_proj(mem_latent)

        encoder_outputs.last_hidden_state = dec_input_hidden_state

        # 3.0. Generate if outputs is None
        if outputs is None:
            return self.generate(encoder_outputs=encoder_outputs, attention_mask=attention_mask,
                                 max_length=max_length, num_beams=num_beams,
                                 decoder_start_token_id=self.config.bos_token_id, use_cache=True)

        # 3.1.  decoder
        decoder_outputs = self.decoder(
            input_ids=outputs['input_ids'],
            attention_mask=outputs['attention_mask'],
            encoder_hidden_states=dec_input_hidden_state,
            encoder_attention_mask=attention_mask,
            use_cache=self.config.use_cache,
            return_dict=self.config.use_return_dict,
        )

        sequence_output = decoder_outputs.last_hidden_state[:, :-1, :]  # b,L_out-1,d
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        # 4. Language modeling task
        lm_logits = self.lm_head(sequence_output)  # b,L_out-1,V
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
        lm_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), (outputs['input_ids'][:, 1:]).reshape(-1))

        if mmd_weight > 0:
            prior_samples = torch.randn_like(mem_latent)
            mmd_loss = self.compute_mmd(mem_latent, prior_samples)
        else:
            mmd_loss = 0

        return AutoencoderOutput(
            lm_logits=lm_logits,
            lm_loss=lm_loss,
            mmd_loss=mmd_loss
        )


    def get_latent_outputs(self, input_ids, attention_mask):

        device = input_ids.device
        origin_input_embeds = self.shared(input_ids)  # b,L,d
        origin_attention_mask = attention_mask
        bsz, L = input_ids.size()

        input_embeds = self.memory_emb * torch.ones((bsz, self.mem_size, 1), device=device)
        attention_mask = torch.ones((bsz, self.mem_size), device=device).int()

        input_embeds = torch.cat((input_embeds, origin_input_embeds), dim=1)  # b,L+L_mem,d
        attention_mask = torch.cat((attention_mask, origin_attention_mask), dim=-1)

        # 2. encoder
        encoder_outputs = self.encoder(input_embeds, attention_mask)

        mem_last_hidden_state = encoder_outputs.last_hidden_state[:, :self.mem_size, :]
        #attention_mask = attention_mask[:, :self.mem_size]

        mem_latent = self.enc_output_proj(mem_last_hidden_state)

        #dec_input_hidden_state = self.dec_input_proj(mem_latent)

        return mem_latent


    def get_decoder_input(self, latent):

        return self.dec_input_proj(latent)