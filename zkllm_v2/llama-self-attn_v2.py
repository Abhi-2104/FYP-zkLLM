import os, sys
import argparse
import torch
import numpy as np
import gc
import subprocess

parser = argparse.ArgumentParser(description='LLaMa-2 Self-Attention V2 - Proof Generation')
parser.add_argument('model_size', type=int, choices = [7, 13], help='The size of the model to use. Default is 13')
parser.add_argument('layer', type=int, help='The layer to use for self-attn')
parser.add_argument('seq_len', type=int, help='The sequence length to use for self-attn')
parser.add_argument('--input_file', required = True, type=str, help='The input file to use for self-attn (output from input rmsnorm)')
parser.add_argument('--output_file', default = 'llama-self-attn-output.bin', type=str, help='The output file to use for self-attn')
parser.add_argument('--workdir', type=str, default=None, help='Work directory for model artifacts and proofs')
parser.add_argument('--precomputed', action='store_true', help='Use precomputed parameters (skip model loading)')
parser.add_argument('--embed_dim', type=int, default=None, help='Embedding dimension (required with --precomputed)')
parser.add_argument('--num_heads', type=int, default=None, help='Number of attention heads (optional with --precomputed)')

from transformers import AutoTokenizer, AutoModelForCausalLM
import fileio_utils


if __name__ == '__main__':
    compilation_error = os.system('make -f Makefile_v2 self-attn_v2')
    if compilation_error:
        print("Error compiling self-attn_v2")
        exit(1)
    
    args = parser.parse_args()

    if args.precomputed:
        embed_dim = args.embed_dim
        if embed_dim is None:
            print("Error: --embed_dim is required with --precomputed")
            exit(1)
        if args.num_heads:
            num_heads = args.num_heads
        else:
            num_heads = embed_dim // 128 if embed_dim % 128 == 0 else 32
    else:
        model_card = f"meta-llama/Llama-2-{args.model_size}b-hf"
        model = AutoModelForCausalLM.from_pretrained(model_card, local_files_only = True, cache_dir = "./model-storage")
        layer = model.model.layers[args.layer].self_attn
        embed_dim = layer.q_proj.in_features
        num_heads = model.config.num_attention_heads
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
    
    # Verify input file exists
    if not os.path.isfile(args.input_file):
        print(f"Error: Input file {args.input_file} not found")
        print("Please run rmsnorm_v2 first to generate the input")
        exit(1)
    
    workdir = args.workdir or f'./zkllm-workdir/Llama-2-{args.model_size}b'
    os.makedirs(workdir, exist_ok=True)
    layer_prefix = f'layer-{args.layer}'

    # Match CUDA binary interface: input seq_len embed_dim workdir layer_prefix output [num_heads]
    cmd = [
        './self-attn_v2',
        str(args.input_file),
        str(args.seq_len),
        str(embed_dim),
        str(workdir),
        str(layer_prefix),
        str(args.output_file),
        str(num_heads)
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: self-attn_v2 failed with exit code {e.returncode}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        exit(1)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
