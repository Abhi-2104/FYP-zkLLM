import os, sys
import argparse
import os, sys
import argparse
import gc

parser = argparse.ArgumentParser(description='LLaMa-2 Self-Attention')
parser.add_argument('model_size', type=int, choices = [7, 13], help='The size of the model to use. Default is 13')
parser.add_argument('layer', type=int, help='The layer to use for rmsnorm')
parser.add_argument('which', type=str, choices=['input', 'post_attention'], help='To use the input norm or the post-attention norm')
parser.add_argument('seq_len', type=int, help='The sequence length to use for rmsnorm')
parser.add_argument('--input_file', required = True, type=str, help='The input file to use for rmsnorm')
parser.add_argument('--output_file', default = 'llama-rmsnorm-output.bin', type=str, help='The output file to use for rmsnorm')
parser.add_argument('--workdir', type=str, default=None, help='Work directory for model artifacts and proofs')
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
        layer = getattr(model.model.layers[args.layer], f'{args.which}_layernorm')
        (embed_dim, ) = layer.weight.shape
        variance_epsilon = layer.variance_epsilon
        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    import torch
    import numpy as np

    if not os.path.isfile(args.input_file):
        temp_X = torch.randn(args.seq_len, embed_dim, device = 0)
        fileio_utils.save_int(temp_X, 1 << 16, args.input_file)
    else:
        num_elems = os.path.getsize(args.input_file) // 4
        if num_elems % embed_dim != 0:
            raise ValueError(f"Input size {num_elems} not divisible by embed_dim {embed_dim}")
        detected_seq = num_elems // embed_dim
        if detected_seq < args.seq_len:
            print(f"Auto-detected seq_len={detected_seq} from input file (lower than requested {args.seq_len}); using {detected_seq}.")
            args.seq_len = detected_seq
        elif detected_seq > args.seq_len:
            print(f"Input has {detected_seq} tokens; truncating to configured seq_len={args.seq_len} for proof generation.")
    X_int = np.fromfile(args.input_file, dtype=np.int32, count=args.seq_len * embed_dim)
    X = torch.tensor(X_int.reshape(args.seq_len, embed_dim), device = 0, dtype = float) / (1 << 16)
    rms_inv = 1 / torch.sqrt(torch.mean(X ** 2, dim = 1) + variance_epsilon)

    workdir = args.workdir or f'./zkllm-workdir/Llama-2-{args.model_size}b'
    os.makedirs(workdir, exist_ok=True)
    layer_prefix = f'layer-{args.layer}'
    # Save rms_inv to a permanent per-layer, per-type file.
    rms_inv_file = f'{workdir}/{layer_prefix}-{args.which}-rms_inv.bin'
    fileio_utils.save_int(rms_inv, 1 << 16, rms_inv_file)

    # Free Python tensors before launching CUDA proof binary.
    del X, rms_inv
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    ret = os.system(f'./rmsnorm_v2 {args.which} {args.input_file} {args.seq_len} {embed_dim} {workdir} {layer_prefix} {args.output_file} {rms_inv_file}')

    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass

    if ret != 0:
        print("\n❌ RMSNorm v2 proof generation failed!")
        exit(1)
    else:
        print("\n✅ RMSNorm v2 proof generation completed successfully!")