import os, sys
import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser(description='LLaMa-2 PPGen')
parser.add_argument('model_size', type=int, choices = [7, 13], help='The size of the model to use. Default is 13')
parser.add_argument('log_scaling_factor', type=int, help='The log scaling factor to use. Default is 16')
parser.add_argument(
    '--delete-int-after-commit',
    action='store_true',
    help='Delete generated *-int.bin files after committing (disabled by default because precomputed proofs need them).'
)

from transformers import AutoTokenizer, AutoModelForCausalLM

def save_weight_int(int_weight: torch.Tensor, path):
    if path[-4:] != '.bin':
        raise ValueError('Path must end with .bin')
    int_weight.cpu().detach().numpy().astype(np.int32).tofile(path)


if __name__ == '__main__':
    args = parser.parse_args()
    model_card = f"meta-llama/Llama-2-{args.model_size}b-hf"
    scaling_factor = 1 << args.log_scaling_factor
    tokenizer = AutoTokenizer.from_pretrained(model_card, local_files_only = True, cache_dir = "./model-storage")
    model = AutoModelForCausalLM.from_pretrained(model_card, local_files_only = True, cache_dir = "./model-storage")

    os.makedirs(f"./zkllm-workdir/Llama-2-{args.model_size}b", exist_ok = True)

    for i, layer in enumerate(model.model.layers):
        for j, w in layer.named_parameters():
            if len(w.shape) == 2:
                w_orig = w.float().T
            else:
                w_orig = w.float()
            w_out = torch.round(w_orig * scaling_factor).to(torch.int32)
            print(f'Max difference of Layer {i}, {j}: {((w_out / scaling_factor) - w_orig).abs().max().item()}')
            pp_path = f"./zkllm-workdir/Llama-2-{args.model_size}b/{j}-pp.bin"
            int_bin_path = f"./zkllm-workdir/Llama-2-{args.model_size}b/layer-{i}-{j}-int.bin"
            commitment_path = f"./zkllm-workdir/Llama-2-{args.model_size}b/layer-{i}-{j}-commitment.bin"
            commitment_exists = os.path.exists(commitment_path) and os.path.getsize(commitment_path) > 0
            int_exists = os.path.exists(int_bin_path) and os.path.getsize(int_bin_path) > 0

            # Always ensure int weights exist for precomputed proof generation.
            if not int_exists:
                save_weight_int(w_out, int_bin_path)
                int_exists = True

            # If commitment already exists, don't recompute it; int regeneration above is enough.
            if commitment_exists:
                print(f"Skipping commitment for {j} - already committed")
                continue

            if len(w_out.shape) == 2:
                ret = os.system(f'./commit-param {pp_path} {int_bin_path} {commitment_path} {w_out.shape[0]} {w_out.shape[1]}')
            else:
                ret = os.system(f'./commit-param {pp_path} {int_bin_path} {commitment_path} {w_out.shape[0]} 1')

            if ret != 0:
                print(f"FAILED on layer {i} {j} with exit code {ret}")
                sys.exit(1)

            if args.delete_int_after_commit and os.path.exists(int_bin_path):
                os.remove(int_bin_path)