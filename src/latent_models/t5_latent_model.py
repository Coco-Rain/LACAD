import torch
import torch.nn as nn

from dataclasses import dataclass

from transformers import T5ForConditionalGeneration, MT5ForConditionalGeneration

from latent_models.fusion_model import Multi_model
from latent_models.gcn_model import GCNEncoder
from latent_models.perceiver_ae import PerceiverAutoEncoder
from einops import rearrange


class T5ForConditionalGenerationLatent(T5ForConditionalGeneration):
    def __init__(self, config, num_encoder_latents, num_decoder_latents, dim_ae, num_layers=2, l2_normalize_latents=False):
        super().__init__(config)
        self.num_encoder_latents = num_encoder_latents
        self.dim_ae = dim_ae
        self.l2_normalize_latents = l2_normalize_latents
        self.perceiver_ae = PerceiverAutoEncoder(dim_lm=config.d_model, num_encoder_latents=num_encoder_latents, num_decoder_latents=num_decoder_latents, dim_ae=dim_ae, depth=num_layers, transformer_decoder=True, l2_normalize_latents=l2_normalize_latents)

    def get_diffusion_latent(self, encoder_outputs, attention_mask):
        hidden_state = encoder_outputs[0]
        latent = self.perceiver_ae.encode(hidden_state, attention_mask.bool())
        return latent
        
    def get_decoder_input(self, diffusion_latent):
        return self.perceiver_ae.decode(diffusion_latent)
    
    # Map encoder outputs to decoder inputs
    def encoder_output_to_decoder_input(self, encoder_outputs, attention_mask):
        diffusion_latent = self.get_diffusion_latent(encoder_outputs, attention_mask)
            
        encoder_outputs['last_hidden_state'] = self.get_decoder_input(diffusion_latent)
        
        return encoder_outputs


class MT5ForConditionalGenerationLatent(MT5ForConditionalGeneration):
    def __init__(self, config, num_encoder_latents, num_decoder_latents, dim_ae, num_layers=2, l2_normalize_latents=False):
        super().__init__(config)
        self.num_encoder_latents = num_encoder_latents
        self.dim_ae = dim_ae
        self.l2_normalize_latents = l2_normalize_latents

        self.perceiver_ae = PerceiverAutoEncoder(dim_lm=config.d_model, num_encoder_latents=num_encoder_latents, num_decoder_latents=num_decoder_latents, dim_ae=dim_ae, depth=num_layers, transformer_decoder=True, l2_normalize_latents=l2_normalize_latents, max_seq_len=192)

    def get_diffusion_latent(self, encoder_outputs, attention_mask):
        hidden_state = encoder_outputs[0]
        latent = self.perceiver_ae.encode(hidden_state, attention_mask.bool())
        return latent
        
    def get_decoder_input(self, diffusion_latent):
        return self.perceiver_ae.decode(diffusion_latent)
    
    # Map encoder outputs to decoder inputs
    def encoder_output_to_decoder_input(self, encoder_outputs, attention_mask):
        diffusion_latent = self.get_diffusion_latent(encoder_outputs, attention_mask)
            
        encoder_outputs['last_hidden_state'] = self.get_decoder_input(diffusion_latent)
        
        return encoder_outputs

class T5ForConditionalGenerationLatent(T5ForConditionalGeneration):
    def __init__(self, config, args, num_encoder_latents, num_decoder_latents, dim_ae, num_layers=2,
                 l2_normalize_latents=False):
        super().__init__(config)
        self.num_encoder_latents = num_encoder_latents
        self.dim_ae = dim_ae
        self.l2_normalize_latents = l2_normalize_latents
        #self.global_context_proj = nn.Linear(config.d_model, dim_ae)
        #self.context_to_modulation = nn.Linear(config.d_model, dim_ae * 2)
        self.gcn_model = GCNEncoder(nfeat=config.d_model, nhid=config.d_model, nout=config.d_model, d_model=config.d_model, batch_size=args.train_batch_size,
                       dropout=0, d_k=64, d_v=64, d_ff=2048, n_heads=8, n_layers=1, device=args.device)

        self.mutil_fusion = Multi_model(args.max_ast_len, args.max_seq_len, d_k=64)
        self.perceiver_ae = PerceiverAutoEncoder(dim_lm=config.d_model, num_encoder_latents=num_encoder_latents,
                                                 num_decoder_latents=num_decoder_latents, dim_ae=dim_ae,
                                                 depth=num_layers, transformer_decoder=True,
                                                 l2_normalize_latents=l2_normalize_latents, max_seq_len=128)

    def get_diffusion_latent(self, encoder_outputs, attention_mask, fusion):
        hidden_state = encoder_outputs[0]
        fusion_attention_mask = torch.ones_like(fusion[:, :, 0])
        layers = [{'input':hidden_state, 'mask': attention_mask.bool()},
                  {'input': fusion, 'mask': fusion_attention_mask.bool()}]

        latent = self.perceiver_ae.encode(layers)
        #fused_latents = latent * (gamma.unsqueeze(1) + 1) + beta.unsqueeze(1)
        return latent

    def get_decoder_input(self, diffusion_latent):
        return self.perceiver_ae.decode(diffusion_latent)

    # Map encoder outputs to decoder inputs
    def encoder_output_to_decoder_input(self, encoder_outputs, attention_mask, fusion):
        diffusion_latent = self.get_diffusion_latent(encoder_outputs, attention_mask, fusion)

        encoder_outputs['last_hidden_state'] = self.get_decoder_input(diffusion_latent)

        return encoder_outputs