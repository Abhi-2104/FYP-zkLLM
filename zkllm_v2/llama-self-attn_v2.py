import os, sys
import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser(description='LLaMa-2 Self-Attention V2 - Proof Generation')
parser.add_argument('model_size', type=int, choices = [7, 13], help='The size of the model to use. Default is 13')
parser.add_argument('layer', type=int, help='The layer to use for self-attn')
parser.add_argument('seq_len', type=int, help='The sequence length to use for self-attn')
parser.add_argument('--input_file', required = True, type=str, help='The input file to use for self-attn (output from input rmsnorm)')
parser.add_argument('--output_file', default = 'llama-self-attn-output.bin', type=str, help='The output file to use for self-attn')

from transformers import AutoTokenizer, AutoModelForCausalLM
import fileio_utils


if __name__ == '__main__':
    compilation_error = os.system('make -f Makefile_v2 self-attn_v2')
    if compilation_error:
        print("Error compiling self-attn_v2")
        exit(1)
    
    args = parser.parse_args()
    model_card = f"meta-llama/Llama-2-{args.model_size}b-hf"

    model = AutoModelForCausalLM.from_pretrained(model_card, local_files_only = True, cache_dir = "./model-storage")
    layer = model.model.layers[args.layer].self_attn
    embed_dim = layer.q_proj.in_features
    
    # Verify input file exists
    if not os.path.isfile(args.input_file):
        print(f"Error: Input file {args.input_file} not found")
        print("Please run rmsnorm_v2 first to generate the input")
        exit(1)
    
    workdir = f'./zkllm-workdir/Llama-2-{args.model_size}b'
    layer_prefix = f'layer-{args.layer}'
    
    os.system(f'./self-attn_v2 {args.input_file} {args.seq_len} {embed_dim} {workdir} {layer_prefix} {args.output_file}')
