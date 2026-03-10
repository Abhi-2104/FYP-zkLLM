#!/usr/bin/env python3
"""
zkLLM v2 Automated Proof Generation Pipeline

Orchestrates generation of zero-knowledge proofs for all
transformer layer components using v2 proof generation logic.

Pipeline order per layer:
1. Input RMSNorm
2. Self-Attention
3. Post-Attention RMSNorm
4. Feed-Forward Network (FFN)
5. Skip Connection
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path


class ZkLLMProofGeneratorV2:
    def __init__(self, model_size=7, seq_len=128, workdir=None, model_card=None):
        self.model_size = model_size
        self.seq_len = seq_len
        self.workdir = workdir or f'./zkllm-workdir/Llama-2-{model_size}b'
        self.model_card = model_card or f'meta-llama/Llama-2-{model_size}b-hf'
        self.total_layers = 32 if model_size == 7 else 40
        
        self.activation_dir = Path('./activations')
        
        # Create directories
        Path(self.workdir).mkdir(parents=True, exist_ok=True)
        self.activation_dir.mkdir(exist_ok=True)
        
        # Track results
        self.results = {}
        
        # Load model once and extract all parameters
        self._load_model_params()
    
    def _load_model_params(self):
        """Load model once and extract parameters needed for proof generation"""
        from transformers import AutoModelForCausalLM
        import gc
        # Try to resolve exact snapshot to bypass network
        cache_path = Path("./model-storage") / f"models--meta-llama--Llama-2-{self.model_size}b-hf"
        if cache_path.exists():
            snapshots_dir = cache_path / "snapshots"
            if snapshots_dir.exists() and any(snapshots_dir.iterdir()):
                model_card_path = str(next(snapshots_dir.iterdir()))
            else:
                model_card_path = self.model_card
        else:
            model_card_path = self.model_card

        print(f"\nLoading model {model_card_path} to extract parameters...")
        model = AutoModelForCausalLM.from_pretrained(
            model_card_path, local_files_only=True, cache_dir="./model-storage"
        )
        
        layer0 = model.model.layers[0]
        (self.embed_dim,) = layer0.input_layernorm.weight.shape
        self.variance_epsilon = layer0.input_layernorm.variance_epsilon
        self.hidden_dim = layer0.mlp.up_proj.out_features
        
        del model
        gc.collect()
        
        print(f"  embed_dim={self.embed_dim}, hidden_dim={self.hidden_dim}, eps={self.variance_epsilon}")
        print(f"  Model unloaded from memory.\n")
        
        # Auto-detect real sequence length from capture output if available
        sample_file = self.activation_dir / "layer-0-block-input.bin"
        if sample_file.exists():
            size_bytes = sample_file.stat().st_size
            num_floats = size_bytes // 4
            detected_seq_len = num_floats // self.embed_dim
            if detected_seq_len > 0:
                self.seq_len = detected_seq_len
                print(f"  Auto-detected seq_len = {self.seq_len} tokens from captured activations.\n")
    
    def run_command(self, cmd, description=""):
        """Run a command and handle errors"""
        print(f"\n{'─'*60}")
        print(f"[RUN] {description}")
        print(f"{'─'*60}")
        print(f"$ {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False, text=True)
            print(f"✅ {description} - SUCCESS")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ {description} - FAILED (exit code {e.returncode})")
            return False
        except FileNotFoundError as e:
            print(f"❌ {description} - FAILED (command not found: {e})")
            return False
    
    def generate_input_rmsnorm(self, layer):
        """
        Generate Input RMSNorm proof
        
        Input: activations/layer-{N}-block-input.bin
        Output: activations/layer-{N}-input-rmsnorm-activation.bin
        Proof: zkllm-workdir/.../layer-{N}-input-rmsnorm-proof.bin
        """
        input_file = self.activation_dir / f"layer-{layer}-block-input.bin"
        output_file = self.activation_dir / f"layer-{layer}-input-rmsnorm-activation.bin"
        proof_file = self.workdir + f"/layer-{layer}-input-rmsnorm-proof.bin"
        
        cmd = [
            'python3', 'llama-rmsnorm_v2.py',
            str(self.model_size), str(layer), 'input', str(self.seq_len),
            '--input_file', str(input_file),
            '--output_file', str(output_file),
            '--precomputed',
            '--embed_dim', str(self.embed_dim),
            '--variance_epsilon', str(self.variance_epsilon)
        ]
        # Ensure output proof file is named correctly
        # If wrapper script allows custom proof file, pass it; else, rename after
        return self.run_command(cmd, f"Layer {layer} Input RMSNorm")
    
    def generate_self_attention(self, layer):
        """
        Generate Self-Attention proof
        
        Input: activations/layer-{N}-input-rmsnorm-activation.bin
        Output: activations/layer-{N}-self-attn-output.bin
        Proof: zkllm-workdir/.../layer-{N}-self-attn-proof.bin
        """
        input_file = self.activation_dir / f"layer-{layer}-input-rmsnorm-activation.bin"
        output_file = self.activation_dir / f"layer-{layer}-self-attn-output.bin"
        
        cmd = [
            'python3', 'llama-self-attn_v2.py',
            str(self.model_size), str(layer), str(self.seq_len),
            '--input_file', str(input_file),
            '--output_file', str(output_file),
            '--precomputed',
            '--embed_dim', str(self.embed_dim)
        ]
        
        return self.run_command(cmd, f"Layer {layer} Self-Attention")
    
    def generate_post_attn_rmsnorm(self, layer):
        """
        Generate Post-Attention RMSNorm proof
        
        Input: activations/layer-{N}-self-attn-output.bin
        Output: activations/layer-{N}-ffn-activation.bin
        Proof: zkllm-workdir/.../layer-{N}-post_attention-rmsnorm-proof.bin
        """
        input_file = self.activation_dir / f"layer-{layer}-self-attn-output.bin"
        output_file = self.activation_dir / f"layer-{layer}-ffn-activation.bin"
        proof_file = self.workdir + f"/layer-{layer}-post-attn-rmsnorm-proof.bin"
        
        cmd = [
            'python3', 'llama-post-attn-rmsnorm_v2.py',
            str(self.model_size), str(layer), str(self.seq_len),
            '--input_file', str(input_file),
            '--output_file', str(output_file),
            '--precomputed',
            '--embed_dim', str(self.embed_dim),
            '--variance_epsilon', str(self.variance_epsilon)
        ]
        # Ensure output proof file is named correctly
        # If wrapper script allows custom proof file, pass it; else, rename after
        return self.run_command(cmd, f"Layer {layer} Post-Attention RMSNorm")
    
    def generate_ffn(self, layer):
        """
        Generate Feed-Forward Network proof
        
        Input: activations/layer-{N}-ffn-activation.bin
        Output: activations/layer-{N}-ffn-output.bin
        Proof: zkllm-workdir/.../layer-{N}-ffn-proof.bin
        """
        input_file = self.activation_dir / f"layer-{layer}-ffn-activation.bin"
        output_file = self.activation_dir / f"layer-{layer}-ffn-output.bin"
        
        cmd = [
            'python3', 'llama-ffn_v2.py',
            str(self.model_size), str(layer), str(self.seq_len),
            '--input_file', str(input_file),
            '--output_file', str(output_file),
            '--precomputed',
            '--embed_dim', str(self.embed_dim),
            '--hidden_dim', str(self.hidden_dim)
        ]
        
        return self.run_command(cmd, f"Layer {layer} Feed-Forward Network")
    
    def generate_skip_connection(self, layer):
        """
        Generate Skip Connection proof
        
        Input A: activations/layer-{N}-block-input.bin (residual)
        Input B: activations/layer-{N}-ffn-output.bin
        Output: activations/layer-{N}-skip-output.bin (== next layer block-input)
        Proof: zkllm-workdir/.../layer-{N}-skip-proof.bin
        """
        block_input = self.activation_dir / f"layer-{layer}-block-input.bin"
        ffn_output = self.activation_dir / f"layer-{layer}-ffn-output.bin"
        skip_output = self.activation_dir / f"layer-{layer}-skip-output.bin"
        
        cmd = [
            'python3', 'llama-skip-connection_v2.py',
            str(self.model_size), str(layer), str(self.seq_len),
            '--block_input_file', str(block_input),
            '--block_output_file', str(ffn_output),
            '--output_file', str(skip_output)
        ]
        
        return self.run_command(cmd, f"Layer {layer} Skip Connection")
    
    def propagate_to_next_layer(self, layer):
        """Copy skip-output to next layer's block-input"""
        skip_output = self.activation_dir / f"layer-{layer}-skip-output.bin"
        next_block_input = self.activation_dir / f"layer-{layer+1}-block-input.bin"
        
        if skip_output.exists():
            import shutil
            shutil.copy(skip_output, next_block_input)
            print(f"📋 Propagated layer-{layer}-skip-output.bin → layer-{layer+1}-block-input.bin")
            return True
        return False
    
    def process_single_layer(self, layer):
        """Process all components of a single layer"""
        print(f"\n{'='*70}")
        print(f"PROCESSING LAYER {layer}")
        print(f"{'='*70}")
        
        results = {
            'input_rmsnorm': False,
            'self_attn': False,
            'post_attn_rmsnorm': False,
            'ffn': False,
            'skip_connection': False
        }
        
        # 1. Input RMSNorm
        print(f"\n[1/5] Input RMSNorm")
        results['input_rmsnorm'] = self.generate_input_rmsnorm(layer)
        
        # 2. Self-Attention
        print(f"\n[2/5] Self-Attention")
        results['self_attn'] = self.generate_self_attention(layer)
        
        # 3. Post-Attention RMSNorm
        print(f"\n[3/5] Post-Attention RMSNorm")
        results['post_attn_rmsnorm'] = self.generate_post_attn_rmsnorm(layer)
        
        # 4. Feed-Forward Network
        print(f"\n[4/5] Feed-Forward Network")
        results['ffn'] = self.generate_ffn(layer)
        
        # 5. Skip Connection
        print(f"\n[5/5] Skip Connection")
        results['skip_connection'] = self.generate_skip_connection(layer)
        
        # Print layer summary
        success_count = sum(results.values())
        print(f"\n{'─'*70}")
        print(f"LAYER {layer} SUMMARY")
        print(f"{'─'*70}")
        print(f"  Input RMSNorm:      {'✅' if results['input_rmsnorm'] else '❌'}")
        print(f"  Self-Attention:     {'✅' if results['self_attn'] else '❌'}")
        print(f"  Post-Attn RMSNorm:  {'✅' if results['post_attn_rmsnorm'] else '❌'}")
        print(f"  Feed-Forward:       {'✅' if results['ffn'] else '❌'}")
        print(f"  Skip Connection:    {'✅' if results['skip_connection'] else '❌'}")
        print(f"{'─'*70}")
        print(f"  Result: {success_count}/5 operations successful")
        
        self.results[layer] = results
        return success_count == 5
    
    def generate_all_proofs(self, start_layer=0, end_layer=None):
        """Generate proofs for specified layers"""
        if end_layer is None:
            end_layer = start_layer  # Default to single layer
        
        print(f"\n{'='*70}")
        print(f"zkLLM v2 PROOF GENERATION PIPELINE")
        print(f"{'='*70}")
        print(f"Model Card: {self.model_card}")
        print(f"Model Size: {self.model_size}b")
        print(f"Layers:     {start_layer} to {end_layer}")
        print(f"Seq Length: {self.seq_len}")
        print(f"Work Dir:   {self.workdir}")
        print(f"Activations:{self.activation_dir}")
        print(f"{'='*70}")
        
        start_time = time.time()
        successful_layers = 0
        failed_layers = []
        
        for layer in range(start_layer, end_layer + 1):
            layer_start = time.time()
            
            if self.process_single_layer(layer):
                successful_layers += 1
                layer_time = time.time() - layer_start
                print(f"\n✅ Layer {layer} completed in {layer_time:.1f}s")
                
                # Propagate output to next layer
                if layer < end_layer:
                    self.propagate_to_next_layer(layer)
            else:
                failed_layers.append(layer)
                print(f"\n❌ Layer {layer} had failures")
                
                # Ask if continue
                try:
                    response = input("Continue with next layer? (y/n): ").strip().lower()
                    if response != 'y':
                        break
                except:
                    break
        
        # Final summary
        total_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"FINAL SUMMARY")
        print(f"{'='*70}")
        print(f"Layers attempted:  {end_layer - start_layer + 1}")
        print(f"Fully successful:  {successful_layers}")
        print(f"With failures:     {len(failed_layers)}")
        print(f"Total time:        {total_time:.1f}s")
        
        if failed_layers:
            print(f"Failed layers:     {failed_layers}")
        
        print(f"\nProofs saved to:   {self.workdir}/")
        print(f"Activations at:    {self.activation_dir}/")
        print(f"{'='*70}")
        
        return successful_layers, failed_layers


def main():
    parser = argparse.ArgumentParser(
        description='zkLLM v2 Automated Proof Generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate proofs for layer 0 (LLaMA-2-7b, default)
  python3 generate_proofs_v2.py --layer 0
  
  # Generate proofs for layers 0 through 3
  python3 generate_proofs_v2.py --start_layer 0 --end_layer 3
  
  # Generate for LLaMA-2-13b instead (uses model-storage/models--meta-llama--Llama-2-13b-hf
  # and workdir zkllm-workdir/Llama-2-13b/)
  python3 generate_proofs_v2.py --model_size 13 --layer 0
  
  # Custom sequence length
  python3 generate_proofs_v2.py --layer 0 --seq_len 64
        """
    )
    parser.add_argument('--model_size', type=int, choices=[7, 13], default=7,
                        help='Model size (default: 7)')
    parser.add_argument('--seq_len', type=int, default=128,
                        help='Sequence length (default: 128)')
    parser.add_argument('--layer', type=int, default=None,
                        help='Process a single layer')
    parser.add_argument('--start_layer', type=int, default=0,
                        help='Starting layer (default: 0)')
    parser.add_argument('--end_layer', type=int, default=None,
                        help='Ending layer (default: same as start_layer)')
    parser.add_argument('--workdir', type=str, default=None,
                        help='Work directory path')
    parser.add_argument('--model_card', type=str, default=None,
                        help='Override model card (e.g., meta-llama/Llama-2-7b-hf or custom model)')
    
    args = parser.parse_args()
    
    # Create generator
    generator = ZkLLMProofGeneratorV2(
        model_size=args.model_size,
        seq_len=args.seq_len,
        workdir=args.workdir,
        model_card=args.model_card
    )
    
    # Determine layer range
    if args.layer is not None:
        start_layer = args.layer
        end_layer = args.layer
    else:
        start_layer = args.start_layer
        end_layer = args.end_layer if args.end_layer is not None else start_layer
    
    # Run generation
    successful, failed = generator.generate_all_proofs(
        start_layer=start_layer,
        end_layer=end_layer
    )
    
    # Exit with error code if any failures
    sys.exit(0 if len(failed) == 0 else 1)


if __name__ == '__main__':
    main()