import os, sys
import argparse
import torch
import numpy as np
import math

parser = argparse.ArgumentParser(description='LLaMa-2 Self-Attention')
parser.add_argument('model_size', type=int, choices = [7, 13], help='The size of the model to use. Default is 13')
parser.add_argument('layer', type=int, help='The layer to use for self-attn')
parser.add_argument('seq_len', type=int, help='The sequence length to use for self-attn')
parser.add_argument('--input_file', required = True, type=str, help='The input file to use for self-attn')
parser.add_argument('--output_file', default = 'llama-self-attn-output.bin', type=str, help='The output file to use for self-attn')

from transformers import AutoTokenizer, AutoModelForCausalLM
from fileio_utils import *

VALUE_LOGSF = 16
ACCU_LOGSF = 20

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

if __name__ == '__main__':
    compilation_error = os.system('make self-attn')
    if compilation_error:
        print("Error compiling self-attn")
        exit(1)
    args = parser.parse_args()
    model_card = f"meta-llama/Llama-2-{args.model_size}b-hf"

    model = AutoModelForCausalLM.from_pretrained(model_card, local_files_only = True, cache_dir = "./model-storage")
    layer = model.model.layers[args.layer]
    embed_dim = layer.self_attn.q_proj.in_features

    workdir = f'./zkllm-workdir/Llama-2-{args.model_size}b'
    layer_prefix = f'layer-{args.layer}'
    os.system(f'./self-attn linear {args.input_file} {args.seq_len} {embed_dim} {workdir} {layer_prefix} {args.output_file}')

    Q, K, V = load_int('temp_Q.bin').reshape(args.seq_len, embed_dim) / (1 << 16), load_int('temp_K.bin').reshape(args.seq_len, embed_dim) / (1 << 16), load_int('temp_V.bin').reshape(args.seq_len, embed_dim) / (1 << 16)

    num_heads = layer.self_attn.config.num_attention_heads
    head_dim = embed_dim // num_heads

    Q = Q.view(args.seq_len, num_heads, head_dim).transpose(0, 1)
    K = K.view(args.seq_len, num_heads, head_dim).transpose(0, 1)
    V = V.view(args.seq_len, num_heads, head_dim).transpose(0, 1)
    
    rotary_emb = None
    if hasattr(layer.self_attn, 'rotary_emb'):
        rotary_emb = layer.self_attn.rotary_emb
    elif hasattr(model.model, 'rotary_emb'):
        rotary_emb = model.model.rotary_emb
    elif hasattr(model.model, 'embed_positions'):
        rotary_emb = model.model.embed_positions
    
    if rotary_emb is not None:
        try:
            rotary_emb.to(0)
            position_ids = torch.arange(0, args.seq_len, dtype=torch.long, device=0).unsqueeze(0)
            
            try:
                cos, sin = rotary_emb(Q, position_ids)
            except:
                try:
                    dummy_input = torch.randn(1, args.seq_len, embed_dim, device=0)
                    cos, sin = rotary_emb(dummy_input, position_ids)
                except:
                    raise Exception("Rotary embedding API failed")
        except:
            rotary_emb = None
    
    if rotary_emb is None:
        print("Using manual rotary embedding calculation")
        
        base = 10000
        if hasattr(model.config, 'rope_theta'):
            base = model.config.rope_theta
        elif hasattr(model.config, 'rotary_emb_base'):
            base = model.config.rotary_emb_base
            
        dim = head_dim
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=0).float() / dim))
        
        position_ids = torch.arange(args.seq_len, device=0).float()
        freqs = torch.outer(position_ids, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos().unsqueeze(0).expand(num_heads, -1, -1)
        sin = emb.sin().unsqueeze(0).expand(num_heads, -1, -1)

    Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
    
    Q, K = Q.to(torch.float64), K.to(torch.float64)
    
    A_ = Q @ K.transpose(-2, -1)
    A = to_int64(A_, VALUE_LOGSF)

    mask = torch.triu(torch.ones(args.seq_len, args.seq_len, device = 0, dtype = bool), diagonal = 1)

    A -= torch.max(A * ~mask, dim = -1, keepdim = True).values 

    shift = math.sqrt(head_dim) * torch.log((torch.exp((to_float(A, ACCU_LOGSF) / math.sqrt(head_dim))) * ~mask).sum(axis = -1, keepdim = True))
    shift = to_int64(shift, ACCU_LOGSF)
    A -= shift
    attn_output = (torch.exp(to_float(A, ACCU_LOGSF, torch.float64) / math.sqrt(head_dim)).float()) * ~mask

    attn_output = attn_output @ V
    attn_output = fromto_int64(attn_output, VALUE_LOGSF)

    attn_output = attn_output.transpose(0, 1).contiguous()
    attn_output = attn_output.view(args.seq_len, embed_dim)
    attn_output = attn_output.reshape(args.seq_len, embed_dim)
    save_int(attn_output, 1 << 16, 'temp_attn_out.bin') 
    os.system(f'./self-attn attn {args.input_file} {args.seq_len} {embed_dim} {workdir} {layer_prefix} {args.output_file}')
    os.system('rm ./temp*.bin')