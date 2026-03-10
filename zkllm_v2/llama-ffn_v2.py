import os, sys
import argparse
import os, sys
import argparse

parser = argparse.ArgumentParser(description='LLaMa-2 FFN V2 - Proof Generation')
parser.add_argument('model_size', type=int, choices = [7, 13], help='The size of the model to use. Default is 13')
parser.add_argument('layer', type=int, help='The layer to use for FFN')
parser.add_argument('seq_len', type=int, help='The sequence length to use for FFN')
parser.add_argument('--input_file', required = True, type=str, help='The input file to use for FFN (output from post-attention rmsnorm)')
parser.add_argument('--output_file', default = 'llama-ffn-output.bin', type=str, help='The output file to use for FFN')
parser.add_argument('--precomputed', action='store_true', help='Use precomputed parameters (skip model loading)')
parser.add_argument('--embed_dim', type=int, default=None, help='Embedding dimension (required with --precomputed)')
parser.add_argument('--hidden_dim', type=int, default=None, help='Hidden dimension (required with --precomputed)')

import fileio_utils

def prepare_swiglu(in_range_num_bit = 9, in_prec_num_bit = 12, out_prec_num_bit = 16):
    """Prepare SwiGLU activation lookup table
    
    Parameters adjusted to ensure D % N == 0 where:
    - D = seq_len * hidden_dim (padded to power of 2)
    - N = table size = 2^(in_range_num_bit + in_prec_num_bit)
    
    For seq_len=128, hidden_dim=11008:
    - D = 1,409,024 (pads to 2^21 = 2,097,152)
    - N must divide D, so N <= 2^21
    - in_range_num_bit=9, in_prec_num_bit=12 gives N=2^21 ✓
    """
def prepare_swiglu(in_range_num_bit = 9, in_prec_num_bit = 12, out_prec_num_bit = 16):
    import torch
    if not os.path.exists('swiglu-table.bin'):
        Xs = torch.arange(- (1 << (in_range_num_bit - 1)), 1 << (in_range_num_bit - 1), step = 1 / (1 << in_prec_num_bit), device = 0)
        Ys = Xs * torch.sigmoid(Xs)
        fileio_utils.save_int(Ys, out_prec_num_bit, 'swiglu-table.bin')


if __name__ == '__main__':
    prepare_swiglu()
    
    # Compile FFN v2
    compilation_error = os.system('make -f Makefile_v2 ffn_v2')
    if compilation_error:
        print("Error compiling ffn_v2")
        exit(1)
    
    args = parser.parse_args()

    if args.precomputed:
        embed_dim = args.embed_dim
        hidden_dim = args.hidden_dim
    else:
        import torch
        from transformers import AutoModelForCausalLM
        model_card = f"meta-llama/Llama-2-{args.model_size}b-hf"
        model = AutoModelForCausalLM.from_pretrained(model_card, local_files_only = True, cache_dir = "./model-storage")
        layer = model.model.layers[args.layer]
        embed_dim = layer.mlp.up_proj.in_features
        hidden_dim = layer.mlp.up_proj.out_features
        del model
        import gc
        gc.collect()
    
    # Verify input file exists
    if not os.path.isfile(args.input_file):
        print(f"Error: Input file {args.input_file} not found")
        print("Please run post-attention rmsnorm first to generate the input")
        exit(1)
    
    workdir = f'./zkllm-workdir/Llama-2-{args.model_size}b'
    layer_prefix = f'layer-{args.layer}'
    
    print(f"\n{'='*70}")
    print(f"Running FFN v2 Proof Generation")
    print(f"{'='*70}")
    print(f"Model: Llama-2-{args.model_size}b")
    print(f"Layer: {args.layer}")
    print(f"Seq len: {args.seq_len}")
    print(f"Embed dim: {embed_dim}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Workdir: {workdir}")
    print(f"{'='*70}\n")
    
    ret = os.system(f'./ffn_v2 {args.input_file} {args.seq_len} {embed_dim} {hidden_dim} {workdir} {layer_prefix} {args.output_file}')
    
    # Keep swiglu-table.bin for verifier to use for claimed output verification
    # Don't delete it - verifier needs it
    if ret != 0:
        print("\n❌ FFN v2 proof generation failed!")
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
        exit(1)
    else:
        print("\n✅ FFN v2 proof generation completed successfully!")
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
