import os, sys
import argparse
import os, sys
import argparse

parser = argparse.ArgumentParser(description='LLaMa-2 Skip Connection V2 - Proof Generation')
parser.add_argument('model_size', type=int, choices=[7, 13], help='The size of the model to use')
parser.add_argument('layer', type=int, help='The layer number')
parser.add_argument('seq_len', type=int, help='The sequence length')
parser.add_argument('--block_input_file', required=True, type=str, help='Input of the block (residual)')
parser.add_argument('--block_output_file', required=True, type=str, help='Output of the block (FFN output)')
parser.add_argument('--output_file', default='skip-output.bin', type=str, help='Output of the skip connection')

import fileio_utils


if __name__ == '__main__':
    # Compile skip-connection v2
    compilation_error = os.system('make -f Makefile_v2 skip-connection_v2')
    if compilation_error:
        print("Error compiling skip-connection_v2")
        exit(1)
    
    args = parser.parse_args()
    
    # Verify input files exist
    if not os.path.isfile(args.block_input_file):
        print(f"Error: Block input file {args.block_input_file} not found")
        exit(1)
    
    if not os.path.isfile(args.block_output_file):
        print(f"Error: Block output file {args.block_output_file} not found")
        exit(1)
    
    workdir = f'./zkllm-workdir/Llama-2-{args.model_size}b'
    layer_prefix = f'layer-{args.layer}'
    
    print(f"\n{'='*70}")
    print(f"Running Skip Connection v2 Proof Generation")
    print(f"{'='*70}")
    print(f"Model: Llama-2-{args.model_size}b")
    print(f"Layer: {args.layer}")
    print(f"Seq len: {args.seq_len}")
    print(f"Block input: {args.block_input_file}")
    print(f"Block output: {args.block_output_file}")
    print(f"Output: {args.output_file}")
    print(f"Workdir: {workdir}")
    print(f"{'='*70}\n")
    
    ret = os.system(f'./skip-connection_v2 {args.block_input_file} {args.block_output_file} {workdir} {layer_prefix} {args.output_file}')
    
    if ret != 0:
        print("\n❌ Skip connection v2 proof generation failed!")
        exit(1)
    else:
        print("\n✅ Skip connection v2 proof generation completed successfully!")
