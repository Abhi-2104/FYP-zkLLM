import os, sys
import argparse
import os, sys
import argparse

parser = argparse.ArgumentParser(description='LLaMa-2 Post-Attention RMSNorm V2 - Proof Generation')
parser.add_argument('model_size', type=int, choices = [7, 13], help='The size of the model to use. Default is 13')
parser.add_argument('layer', type=int, help='The layer to use for post-attention rmsnorm')
parser.add_argument('seq_len', type=int, help='The sequence length to use for rmsnorm')
parser.add_argument('--input_file', required = True, type=str, help='The input file to use for post-attention rmsnorm (output from self-attention)')
parser.add_argument('--output_file', default = 'llama-post-attn-rmsnorm-output.bin', type=str, help='The output file to use for post-attention rmsnorm')
parser.add_argument('--precomputed', action='store_true', help='Use precomputed parameters (skip model loading)')
parser.add_argument('--embed_dim', type=int, default=None, help='Embedding dimension (required with --precomputed)')
parser.add_argument('--variance_epsilon', type=float, default=None, help='Variance epsilon (required with --precomputed)')

import fileio_utils


if __name__ == '__main__':
    compilation_error = os.system('make -f Makefile_v2 rmsnorm_v2')
    if compilation_error:
        print("Error compiling rmsnorm_v2")
        exit(1)
    args = parser.parse_args()
    
    if args.precomputed:
        embed_dim = args.embed_dim
        variance_epsilon = args.variance_epsilon
    else:
        import torch
        from transformers import AutoModelForCausalLM
        model_card = f"meta-llama/Llama-2-{args.model_size}b-hf"
        model = AutoModelForCausalLM.from_pretrained(model_card, local_files_only = True, cache_dir = "./model-storage")
        layer = getattr(model.model.layers[0], 'post_attention_layernorm')
        (embed_dim, ) = layer.weight.shape
        variance_epsilon = layer.variance_epsilon
        del model
        import gc
        gc.collect()
    
    import torch
    import numpy as np

    if not os.path.isfile(args.input_file):
        temp_X = torch.randn(args.seq_len, embed_dim, device = 0)
        fileio_utils.save_int(temp_X, 1 << 16, args.input_file)
    X = torch.tensor(np.fromfile(args.input_file, dtype = np.int32).reshape(args.seq_len, embed_dim), device = 0, dtype = float) / (1 << 16)
    rms_inv = 1 / torch.sqrt(torch.mean(X ** 2, dim = 1) + variance_epsilon)
    
    workdir = f'./zkllm-workdir/Llama-2-{args.model_size}b'
    layer_prefix = f'layer-{args.layer}'
    # Save rms_inv to a permanent per-layer, per-type file so the verifier
    # can always find the correct one regardless of execution order.
    rms_inv_file = f'{workdir}/{layer_prefix}-post_attention-rms_inv.bin'
    fileio_utils.save_int(rms_inv, 1 << 16, rms_inv_file)
    
    ret = os.system(f'./rmsnorm_v2 post_attention {args.input_file} {args.seq_len} {embed_dim} {workdir} {layer_prefix} {args.output_file} {rms_inv_file}')

    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass

    if ret != 0:
        print("\n❌ Post-Attention RMSNorm v2 proof generation failed!")
        exit(1)
    else:
        print("\n✅ Post-Attention RMSNorm v2 proof generation completed successfully!")
