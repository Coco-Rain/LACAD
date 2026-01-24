import re, json
from transformers import AutoTokenizer, PreTrainedTokenizerBase, T5ForConditionalGeneration, AutoModelForCausalLM, MBartTokenizerFast, MT5ForConditionalGeneration
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from transformers.models.mbart.modeling_mbart import MBartForConditionalGeneration

import CONSTANTS as CONSTANTS

from latent_models.bart_latent_model import BARTForConditionalGenerationLatent
from latent_models.t5_latent_model import T5ForConditionalGenerationLatent, MT5ForConditionalGenerationLatent



def get_latent_model(args):
    """
    if 'bart' in args.enc_dec_model:
        local_model_path = 'D:\\Develop\\Diffusion-Model\latent-diffusion-for-language-main\\bart-base'  # Replace with the local path to the model
        config = BartForConditionalGeneration.from_pretrained(local_model_path).config
        lm = BARTForConditionalGenerationLatent.from_pretrained(
            local_model_path, config=config,
            num_encoder_latents=args.num_encoder_latents,
            num_decoder_latents=args.num_decoder_latents,
            dim_ae=args.dim_ae, num_layers=args.num_layers,
            l2_normalize_latents=args.l2_normalize_latents,
            _fast_init=False
        )
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(local_model_path)
    """

    if 'bart' in args.enc_dec_model:
        config = BartForConditionalGeneration.from_pretrained(
            args.enc_dec_model).config
        lm = BARTForConditionalGenerationLatent.from_pretrained(
            args.enc_dec_model, config=config, num_encoder_latents=args.num_encoder_latents,
            num_decoder_latents=args.num_decoder_latents, dim_ae=args.dim_ae, num_layers=args.num_layers,
            l2_normalize_latents=args.l2_normalize_latents, _fast_init=False)
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            args.enc_dec_model)

    elif 't5' in args.enc_dec_model:
        if 'mt5' in args.enc_dec_model:
            config = MT5ForConditionalGeneration.from_pretrained(
                args.enc_dec_model).config
            lm = MT5ForConditionalGenerationLatent.from_pretrained(
                args.enc_dec_model, config=config, num_encoder_latents=args.num_encoder_latents, num_decoder_latents=args.num_decoder_latents, dim_ae=args.dim_ae, num_layers=args.num_layers, l2_normalize_latents=args.l2_normalize_latents, _fast_init=False)
            tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
                args.enc_dec_model)
        elif 'codet5' in args.enc_dec_model:
            config = T5ForConditionalGeneration.from_pretrained(
                args.enc_dec_model).config
            lm = T5ForConditionalGenerationLatent.from_pretrained(
                args.enc_dec_model, config=config, args = args, num_encoder_latents=args.num_encoder_latents,
                num_decoder_latents=args.num_decoder_latents, dim_ae=args.dim_ae, num_layers=args.num_layers,
                l2_normalize_latents=args.l2_normalize_latents, _fast_init=False)
            tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
                args.enc_dec_model)
            try:
                with open('new_vocab_list.json', 'r') as f:
                    # 加载现有的数据
                    all_method_names_list = json.load(f)
            except FileNotFoundError:
                # 如果文件不存在，则初始化为空列表
                all_method_names_list = []
            tokenizer.add_tokens(all_method_names_list)
            lm.resize_token_embeddings(len(tokenizer))
        else:
            config = T5ForConditionalGeneration.from_pretrained(
                args.enc_dec_model).config
            lm = T5ForConditionalGenerationLatent.from_pretrained(
                args.enc_dec_model, config=config, num_encoder_latents=args.num_encoder_latents, num_decoder_latents=args.num_decoder_latents, dim_ae=args.dim_ae, num_layers=args.num_layers, l2_normalize_latents=args.l2_normalize_latents, _fast_init=False)
            tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
                args.enc_dec_model)
    else:
        print("Unsupported model")
        raise NotImplementedError
    
    if args.lm_mode == 'ft':
        for (param_name, param) in lm.named_parameters():
            param.requires_grad = True
    elif args.lm_mode == 'freeze':
        for (param_name, param) in lm.named_parameters():
            if re.fullmatch(".*perceiver.*", param_name):
                param.requires_grad = True
                print(f"Trainable: {param_name}")
            elif re.fullmatch(".*mutil_fusion.*", param_name):
                param.requires_grad = True
                print(f"Trainable: {param_name}")
            elif re.fullmatch(".*gcn_model.*", param_name):
                param.requires_grad = True
                print(f"Trainable: {param_name}")
            else:
                param.requires_grad = False
    else:
        raise NotImplementedError
        


        
    return lm, tokenizer, config