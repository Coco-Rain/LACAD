from transformers import T5ForConditionalGeneration, T5Config
from transformers.models.t5.modeling_t5 import T5Attention, T5LayerSelfAttention, T5Block, T5LayerFF, T5Stack
import torch.nn as nn
import torch
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.utils import ModelOutput
from typing import Optional
import torch.nn.functional as F
from dataclasses import dataclass
from torch.nn  import Module

class AutoencoderEncoderAttention(T5Attention):
    def __init__(self, config, dtype, mem_len, num_heads):
        super().__init__(config, has_relative_attention_bias=True)
        self.is_decoder = False
        self.dtype = dtype
        self.mem_len = mem_len
        self.num_heads = num_heads

        # learnable bias for mem↔mem attention
        self.mem_position_bias = nn.Parameter(torch.zeros(self.num_heads, mem_len, mem_len))  # [h, mem_len, mem_len]

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

    def forward(self, hidden_states, mask, dfg_code_links, dfg_dfg_links, ast_code_links, ast_ast_sims,
                position_bias=None, output_attentions=False):
        # hidden_states -> b,L+L_dfg+L_ast,d
        # mask -> b,L+L_dfg+L_ast -> 0 or 1
        # dfg_code_links -> bsz, L_dfg, L -> 0 or 1
        # dfg_dfg_links -> bsz, L_dfg, L_dfg -> 0 or 1
        # ast_code_links -> bsz, L_ast, L -> 0 or 1
        # ast_ast_sims -> bsz, L_ast, L_ast
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
            position_bias[:, :mem_len, :mem_len] = self.mem_position_bias

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
        super().__init__(config,  has_relative_attention_bias=True)
        self.SelfAttention = AutoencoderEncoderAttention(config, dtype, mem_len, num_heads)


    def forward(self, hidden_states, mask, dfg_code_links, dfg_dfg_links, ast_code_links, ast_ast_sims,
                position_bias=None, output_attentions=False):
        # hidden_states -> b,L+L_dfg+L_ast,d
        # mask -> b,L+L_dfg+L_ast -> 0 or 1
        # dfg_code_links -> bsz, L_dfg, L -> 0 or 1
        # dfg_dfg_links -> bsz, L_dfg, L_dfg -> 0 or 1
        # ast_code_links -> bsz, L_ast, L -> 0 or 1
        # ast_ast_sims -> bsz, L_ast, L_ast
        # position_bias -> bsz, h, L', L' -> code-code and leaf-leaf positional attention and all masks

        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(normed_hidden_states, mask,
                                              dfg_code_links, dfg_dfg_links, ast_code_links, ast_ast_sims,
                                              position_bias, output_attentions)
        hidden_states = hidden_states + self.dropout(attention_output[0])
        return (hidden_states,) + attention_output[1:]  # (bL'd) (bhL'L') (bhL'L')


class AutoencoderEncoderBlock(T5Block):
    def __init__(self, config, dtype, mem_len, num_heads):
        super().__init__(config, True)
        self.layer = nn.ModuleList()
        self.layer.append(AutoencoderEncoderLayerSelfAttention(config, dtype, mem_len, num_heads))
        self.layer.append(T5LayerFF(config))

    def forward(self, hidden_states, mask, dfg_code_links, dfg_dfg_links, ast_code_links, ast_ast_sims,
                position_bias=None, output_attentions=False):
        # hidden_states -> b,L+L_dfg+L_ast,d
        # mask -> b,L+L_dfg+L_ast -> 0 or 1
        # dfg_code_links -> bsz, L_dfg, L -> 0 or 1
        # dfg_dfg_links -> bsz, L_dfg, L_dfg -> 0 or 1
        # ast_code_links -> bsz, L_ast, L -> 0 or 1
        # ast_ast_sims -> bsz, L_ast, L_ast
        # position_bias -> bsz, h, L', L' -> code-code and leaf-leaf positional attention and all masks

        self_attention_outputs = self.layer[0](hidden_states, mask, dfg_code_links, dfg_dfg_links,
                                               ast_code_links, ast_ast_sims,
                                               position_bias=position_bias, output_attentions=output_attentions)
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
        self.block = nn.ModuleList([AutoencoderEncoderBlock(config, self.dtype, mem_len, num_heads) for i in range(config.num_layers)])

    def forward(self, inputs_embeds, attention_mask, dfg_code_links, dfg_dfg_links, ast_code_links, ast_ast_sims,
                output_attentions=False, output_hidden_states=False):
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

            layer_outputs = layer_module(hidden_states, attention_mask, dfg_code_links, dfg_dfg_links,
                                         ast_code_links, ast_ast_sims,
                                         position_bias=position_bias, output_attentions=output_attentions)
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


class AutoencoderForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, args):
        config = T5Config.from_pretrained('Salesforce/codet5-base')
        if args.model_size == 'small':
            config.d_model, config.d_kv, config.d_ff = 256, 32, 1024
            config.num_layers, config.num_decoder_layers, config.num_heads = 5, 5, 8
        super().__init__(config)
        factor = config.initializer_factor

        self.mem_size = args.mem_slots_len

        # memory encoder weights
        self.memory_emb = nn.Parameter(torch.empty((1, 1, self.config.d_model)), requires_grad=True)
        self.memory_emb.data.normal_(mean=0.0, std=factor * 1.0)

        # encoder
        self.encoder = AutoencoderEncoderStack(config, self.mem_size, config.num_heads)

        self.eps = 1e-10


        # load pretrained weights
        if args.model_size != 'small':
            args.logger.write('\nLoading StructCoder weights from CodeT5.')
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
            args.logger.write('Could not load weights "' + str(not_found) + '" from pretrained CodeT5.')

    # to support generate()
    def forward2(self, encoder_outputs, attention_mask, decoder_input_ids, past_key_values):
        decoder_outputs = self.decoder(input_ids=decoder_input_ids,
                                        encoder_hidden_states=encoder_outputs.last_hidden_state,
                                        encoder_attention_mask=attention_mask,
                                        past_key_values=past_key_values,
                                        use_cache=self.config.use_cache
                                        )
        sequence_output = decoder_outputs[0] # b,L_out,d
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim**-0.5)
        lm_logits = self.lm_head(sequence_output) #b,L_out,V
        return Seq2SeqLMOutput(logits=lm_logits,
                               past_key_values=decoder_outputs.past_key_values,
                               decoder_hidden_states=decoder_outputs.hidden_states)



    # TODO: Add support for output_attentions, output_hidden_states
    def forward(self, inputs=None, outputs=None,
                encoder_outputs=None, attention_mask=None, decoder_input_ids=None,
                past_key_values=None, max_length=None, num_beams=None, **kwargs):
        # inputs['input_ids'] -> bsz, L
        # inputs['attention_mask'] -> bsz, L

        # remaining inputs are for forward2
        if encoder_outputs is not None:
            return self.forward2(encoder_outputs, attention_mask, decoder_input_ids, past_key_values)

        # 1. Input embedding
        device = inputs['input_ids'].device
        origin_input_embeds = self.shared(inputs['input_ids'])  # b,L,d
        origin_attention_mask = inputs['attention_mask']
        L_dfg, L_ast =  0, 0
        bsz, L = inputs['input_ids'].size()

        input_embeds = self.memory_emb * torch.ones((bsz, self.mem_size, 1), device=device)
        attention_mask = torch.ones((bsz, self.mem_size), device=device).int()


        input_embeds = torch.cat((input_embeds, origin_input_embeds), dim=1)  # b,L+L_mem,d
        attention_mask = torch.cat((attention_mask, origin_attention_mask), dim=-1)


        """
        # add dfg nodes to input_embeds and attention_mask
        if inputs.get('num_dfg_nodes', None) is not None:
            bsz, L_dfg, _ = inputs['dfg_code_links'].size()
            dfg_embeds = self.dfg_node_emb * torch.ones((bsz, L_dfg, 1),
                                                        device=device)  # b,L_dfg,d     所有 DFG 节点用同一个向量（或初始一样，后续训练微调）
            input_embeds = torch.cat((input_embeds, dfg_embeds), dim=1)  # b,L+L_dfg,d
            dfg_attention_mask = (
                        torch.arange(L_dfg, device=device)[None, :] < inputs['num_dfg_nodes'][:, None]).int()  # b,L_dfg
            attention_mask = torch.cat((attention_mask, dfg_attention_mask), dim=-1)  # b,L+L_dfg

        # add ast leaves to input_embeds and attention_mask
        if inputs.get('ast_paths', None) is not None:
            L_ast = inputs['ast_paths'].size()[1]
            ast_paths_mask = (inputs['ast_paths'] >= 0)  # b,L_ast,max_depth
            ast_paths = torch.clip(inputs['ast_paths'], 0)  # b,L_ast,max_depth    # 替换负数为 0，便于后续计算
            ast_leaf_embeds = self.ast_type_emb(ast_paths) + self.ast_depth_emb  # b,L_ast,max_depth,d
            ast_leaf_embeds = (ast_leaf_embeds * ast_paths_mask[:, :, :, None]).sum(
                dim=2)  # b,L_ast,d   把每条路径中有效的节点类型嵌入 + 深度嵌入 逐点相加
            input_embeds = torch.cat((input_embeds, ast_leaf_embeds), dim=1)  # b,L+L_dfg+L_ast,d    拼接到原始 input_embeds
            ast_attention_mask = torch.clip(ast_paths_mask.sum(dim=-1), 0,
                                            1)  # b,L_ast   构造 AST 部分的注意力掩码，对每个 AST 节点，只要其路径中有任意一个非 padding 节点，就算是有效节点（mask 为 1）
            attention_mask = torch.cat((attention_mask, ast_attention_mask), dim=-1)  # b,L+L_dfg+L_ast
        
        """
        # 2. StructCoder encoder
        encoder_outputs = self.encoder(input_embeds, attention_mask, inputs.get('dfg_code_links', None),
                                       inputs.get('dfg_dfg_links', None), inputs.get('ast_code_links', None),
                                       inputs.get('ast_ast_sims', None))


        # 3.0. Generate if outputs is None
        if outputs is None:
            return self.generate(encoder_outputs=encoder_outputs, attention_mask=attention_mask,
                                 max_length=max_length, num_beams=num_beams,
                                 decoder_start_token_id=self.config.bos_token_id, use_cache=True)

        mem_last_hidden_state = encoder_outputs.last_hidden_state[:, :self.mem_size, :]
        decoder_attention_mask = attention_mask[:, :self.mem_size]
        # 3.1.  decoder
        decoder_outputs = self.decoder(
            input_ids=outputs['input_ids'],
            attention_mask=outputs['attention_mask'],
            encoder_hidden_states=mem_last_hidden_state,  # 截取前 mem_size 个token
            encoder_attention_mask=decoder_attention_mask,  # 确保attention_mask也对应截取
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

        """
        
        # 5. DFG links prediction task
        if 'dfg_dfg_links' in outputs:
            dfg_hidden = sequence_output[:, :, :self.dfg_bits]
            hidden1 = self.dfg_weight1(dfg_hidden)  # b,L-1,d'
            hidden2 = self.dfg_weight2(dfg_hidden)  # b,L-1,d'
            dfg_logits = torch.bmm(hidden1, hidden2.permute([0, 2, 1]).contiguous()) + self.dfg_b1(dfg_hidden) + \
                         self.dfg_b2(dfg_hidden).permute(0, 2,
                                                         1) + self.dfg_b3  # b,L-1,L-1       计算所有节点对的交互分数（矩阵乘法）  dfg_b1 和 dfg_b2 分别生成节点级的偏置，dfg_b3 是全局标量偏置。
            dfg_loss = F.binary_cross_entropy_with_logits(dfg_logits, outputs['dfg_dfg_links'][:, 1:, 1:],
                                                          reduction='none')  # -log p(correct_class)
            is_pos = outputs['dfg_dfg_links'][:, 1:, 1:] == 1  # 正样本（连接存在）掩码
            is_neg = outputs['dfg_dfg_links'][:, 1:, 1:] == 0  # 负样本（连接存在）掩码
            dfg_loss = (dfg_loss * is_pos).sum() / (2 * torch.clamp(is_pos.sum(), min=1)) \
                       + (dfg_loss * is_neg).sum() / (2 * torch.clamp(is_neg.sum(),
                                                                      min=1))  # 正样本（连接存在）和负样本（无连接）的损失分别计算均值    最终损失为两者均值之和，确保正负样本权重相等。 平衡优化：DFG 连接通常是稀疏的（负样本远多于正样本），平衡损失避免模型偏向预测负类。
        else:
            dfg_logits, dfg_loss = None, None

        # 6. AST paths prediction task
        if 'ast_paths' in outputs:
            ast_hidden = sequence_output[:, :, self.dfg_bits:self.dfg_bits + self.ast_path_bits]
            ast_logits = self.ast_path_head(ast_hidden)  # b,L-1,max_depth*num_node_types
            ast_logits = ast_logits.view(-1, ast_logits.size()[1], self.args.max_ast_depth,
                                         self.args.num_node_types)  # b, L-1, max_depth, num_node_types
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            ast_loss = loss_fct(ast_logits.view(-1, ast_logits.size(-1)),  # b*(L-1)*max_depth, num_node_types
                                (outputs['ast_paths'][:, 1:, :]).reshape(-1).long())  # b*(L-1)*max_depth,
        else:
            ast_logits, ast_loss = None, None
        """

        return AutoencoderOutput(
            lm_logits=lm_logits,
            lm_loss=lm_loss,
        )


class PEFTCompatibleWrapper(Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        # 自动代理所有未在包装类中定义的方法到原始模型
        self._proxied_methods = set()

    def __getattr__(self, name):
        # 代理所有未定义的属性访问到原始模型
        if name in self._proxied_methods:
            return object.__getattribute__(self.model, name)
        try:
            attr = getattr(self.model, name)
            self._proxied_methods.add(name)
            return attr
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' and wrapped model have no attribute '{name}'")

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # 转换标准参数到您的自定义参数
        return self.model(
            inputs=input_ids,
            outputs=labels,
            attention_mask=attention_mask,
            **kwargs
        )

    # 确保包装类也能正确进行设备移动
    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)
        return self

        # 代理prepare_inputs_for_generation方法（生成任务必需）

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.model.prepare_inputs_for_generation(*args, **kwargs)