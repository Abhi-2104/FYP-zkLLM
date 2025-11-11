#!/usr/bin/env python3
"""
RMSNorm ZK Proof Verification Demo
Demonstrates end-to-end: Proof Generation → Verification

Usage:
    python demo_verification.py 7 0 input 128 --input_file activations/layer-0-block-input.bin
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print()
    print("╔" + "═" * 70 + "╗")
    print(f"║ {title:^68} ║")
    print("╚" + "═" * 70 + "╝")
    print()

def print_section(title):
    """Print a section header"""
    print()
    print(f"═══ {title} ═══")
    print()

def run_command(cmd, description, show_output=True):
    """Run a command and handle errors"""
    print(f"{description}...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=not show_output
        )
        if not show_output and result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Command failed with exit code {e.returncode}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def show_file_info(filepath, description):
    """Show file information"""
    path = Path(filepath)
    if path.exists():
        size = path.stat().st_size
        if size < 1024:
            size_str = f"{size} bytes"
        elif size < 1024*1024:
            size_str = f"{size/1024:.2f} KB"
        else:
            size_str = f"{size/(1024*1024):.2f} MB"
        print(f"{description}:")
        print(f"  File: {filepath}")
        print(f"  Size: {size_str}")
    else:
        print(f"{description}:")
        print(f"  ⚠ File not found: {filepath}")
    print()

def main():
    parser = argparse.ArgumentParser(
        description='RMSNorm ZK Proof Generation and Verification Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python demo_verification.py 7 0 input 128 --input_file activations/layer-0-block-input.bin
        """
    )
    
    parser.add_argument('model_size', type=int, choices=[7, 13], help='Model size (7 or 13)')
    parser.add_argument('layer', type=int, help='Layer number')
    parser.add_argument('which', choices=['input', 'post_attention'], help='Normalization type')
    parser.add_argument('seq_len', type=int, help='Sequence length')
    parser.add_argument('--input_file', required=True, help='Input activation file')
    parser.add_argument('--output_file', default=None, help='Output file (default: layer-{layer}-rmsnorm-output.bin)')
    
    args = parser.parse_args()
    
    # Configuration
    workdir = f"zkllm-workdir/Llama-2-{args.model_size}b"
    layer_prefix = f"layer-{args.layer}"
    which = args.which
    proof_file = f"{workdir}/{layer_prefix}-rmsnorm-proof.bin"
    input_file = args.input_file
    output_file = args.output_file or f"{layer_prefix}-rmsnorm-output.bin"
    
    print_header("RMSNorm ZK Proof Verification Demo")
    print("End-to-End: Proof Generation → Verification")
    
    # Step 1: Show existing files
    print_section("STEP 1: Show Existing Files")
    
    show_file_info(input_file, "Input Activation")
    show_file_info(
        f"{workdir}/{layer_prefix}-{which}_layernorm.weight-commitment.bin",
        "Model Weight Commitment"
    )
    show_file_info(
        f"{workdir}/{layer_prefix}-{which}_layernorm.weight-int.bin",
        "Model Weight Values"
    )
    
    # Step 2: Generate proof
    print_section("STEP 2: Generate Proof")
    
    cmd = [
        "python3", "llama-rmsnorm_v2.py",
        str(args.model_size), str(args.layer), which, str(args.seq_len),
        "--input_file", input_file,
        "--output_file", output_file
    ]
    
    if not run_command(cmd, "Running RMSNorm proof generation", show_output=True):
        print("\n✗ Proof generation failed!")
        return 1
    
    # Step 3: Show generated proof
    print_section("STEP 3: Show Generated Proof")
    show_file_info(proof_file, "Generated Proof")
    
    # Step 4: Verify proof
    print_section("STEP 4: Verify Proof Against Commitments")
    
    cmd = [
        "./verify_rmsnorm_v2",
        proof_file,
        workdir,
        layer_prefix,
        which
    ]
    
    if not run_command(cmd, "Running verification", show_output=True):
        print("\n✗ Verification failed!")
        return 1
    
    # Summary
    print()
    print_header("DEMO COMPLETED SUCCESSFULLY")
    
    print("What This Demonstrates:")
    print("  ✓ ZK proof generation for RMSNorm computation")
    print("  ✓ Proof serialization to disk (portable proofs)")
    print("  ✓ Separate proof verification against model commitments")
    print("  ✓ Cryptographic commitment scheme (weights stay private)")
    print("  ✓ Production-ready: prover/verifier separation")
    print()
    
    print("Scalability:")
    print("  • Same pattern works for all 32 layers")
    print("  • Extends to self-attention, FFN, full transformer")
    print("  • Batch verification for multiple layers")
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
