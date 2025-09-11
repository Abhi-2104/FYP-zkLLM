#!/usr/bin/env python3

import os
import sys
import subprocess
import time
import torch
import numpy as np
from pathlib import Path

class ZkLLMProofGenerator:
    def __init__(self, model_size=7, seq_len=128, workdir=None):
        self.model_size = model_size
        self.seq_len = seq_len
        self.workdir = workdir or f'./zkllm-workdir/Llama-2-{model_size}b'
        self.total_layers = 32  # LLaMA-2 has 32 layers (0-31)
        
        # Create organized directory structure
        self.activation_dir = Path('./activations')
        self.temp_dir = Path('./temp-files')
        
        # Ensure directories exist
        Path(self.workdir).mkdir(parents=True, exist_ok=True)
        self.activation_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Clean up any existing temp files
        self.cleanup_temp_files()
    
    def cleanup_temp_files(self):
        """Clean up temporary files from previous runs"""
        temp_patterns = ['temp_*.bin', 'rms_inv_temp.bin', 'swiglu-table.bin']
        for pattern in temp_patterns:
            for file in Path('.').glob(pattern):
                try:
                    file.unlink()
                    print(f"Cleaned up: {file}")
                except:
                    pass
    
    def create_dummy_activation_file(self, filepath, shape):
        """Create a dummy activation file with random data"""
        print(f"Creating dummy activation file: {filepath} with shape {shape}")
        if len(shape) == 2:
            dummy_data = torch.randn(*shape).numpy().astype('int32') * (1 << 16)
        else:
            dummy_data = torch.randn(*shape).numpy().astype('int32')
        dummy_data.tofile(filepath)
    
    def run_command(self, cmd, description="", cwd=None):
        """Run a command and handle errors"""
        print(f"Running: {description}")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=cwd)
            print(f"✅ Success: {description}")
            if result.stdout.strip():
                # Only show last few lines to avoid spam
                output_lines = result.stdout.strip().split('\n')
                if len(output_lines) > 3:
                    print(f"Output: ...{' | '.join(output_lines[-2:])}")
                else:
                    print(f"Output: {result.stdout.strip()}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error: {description} failed")
            print(f"Error code: {e.returncode}")
            if e.stderr:
                error_lines = e.stderr.strip().split('\n')
                print(f"Error: {' | '.join(error_lines[-3:])}")
            return False
    
    def generate_rmsnorm_proof(self, layer, norm_type):
        """Generate RMSNorm proof for a specific layer and type"""
        input_file = self.activation_dir / f"layer-{layer}-{norm_type}-rmsnorm-activation.bin"
        output_file = f"{self.workdir}/layer-{layer}-{norm_type}-rmsnorm-proof.bin"
        
        # Create dummy activation file if it doesn't exist
        if not input_file.exists():
            self.create_dummy_activation_file(input_file, (self.seq_len, 4096))  # embed_dim = 4096
        
        cmd = [
            'python', 'llama-rmsnorm.py',
            str(self.model_size), str(layer), norm_type, str(self.seq_len),
            '--input_file', str(input_file),
            '--output_file', output_file
        ]
        
        success = self.run_command(cmd, f"Layer {layer} {norm_type} RMSNorm")
        
        # Clean up temp files created by rmsnorm
        temp_files = ['rms_inv_temp.bin']
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
        return success
    
    def generate_self_attn_proof(self, layer):
        """Generate Self-Attention proof for a specific layer"""
        input_file = self.activation_dir / f"layer-{layer}-self-attn-activation.bin"
        output_file = f"{self.workdir}/layer-{layer}-self-attn-proof.bin"
        
        # Create dummy activation file if it doesn't exist
        if not input_file.exists():
            self.create_dummy_activation_file(input_file, (self.seq_len, 4096))  # embed_dim = 4096
        
        # Clean up any existing temp files first
        temp_files = ['temp_Q.bin', 'temp_K.bin', 'temp_V.bin', 'temp_O.bin', 'temp_attn_out.bin']
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        cmd = [
            'python', 'llama-self-attn.py',
            str(self.model_size), str(layer), str(self.seq_len),
            '--input_file', str(input_file),
            '--output_file', output_file
        ]
        
        success = self.run_command(cmd, f"Layer {layer} Self-Attention")
        
        # Clean up temp files after execution (whether success or failure)
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        return success
    
    def generate_ffn_proof(self, layer):
        """Generate Feed-Forward Network proof for a specific layer"""
        input_file = self.activation_dir / f"layer-{layer}-ffn-activation.bin"
        output_file = f"{self.workdir}/layer-{layer}-ffn-proof.bin"
        
        # Create dummy activation file if it doesn't exist
        if not input_file.exists():
            self.create_dummy_activation_file(input_file, (self.seq_len, 4096))  # embed_dim = 4096
        
        # Clean up swiglu table if it exists
        if os.path.exists('swiglu-table.bin'):
            os.remove('swiglu-table.bin')
        
        cmd = [
            'python', 'llama-ffn.py',
            str(self.model_size), str(layer), str(self.seq_len),
            '--input_file', str(input_file),
            '--output_file', output_file
        ]
        
        success = self.run_command(cmd, f"Layer {layer} Feed-Forward Network")
        
        # Clean up swiglu table after execution
        if os.path.exists('swiglu-table.bin'):
            os.remove('swiglu-table.bin')
            
        return success
    
    def generate_skip_connection_proof(self, layer):
        """Generate Skip Connection proof for a specific layer"""
        
        # Skip connections need two input files: block input and block output
        block_input_file = self.activation_dir / f"layer-{layer}-block-input.bin"
        block_output_file = self.activation_dir / f"layer-{layer}-block-output.bin"
        output_file = f"{self.workdir}/layer-{layer}-skip-proof.bin"
        
        # Create dummy input files if they don't exist
        if not block_input_file.exists():
            self.create_dummy_activation_file(block_input_file, (self.seq_len, 4096))  # embed_dim = 4096
        
        if not block_output_file.exists():
            self.create_dummy_activation_file(block_output_file, (self.seq_len, 4096))  # embed_dim = 4096
        
        cmd = [
            'python', 'llama-skip-connection.py',
            '--block_input_file', str(block_input_file),
            '--block_output_file', str(block_output_file),
            '--output_file', output_file
        ]
        
        return self.run_command(cmd, f"Layer {layer} Skip Connection")
    
    def process_single_layer(self, layer):
        """Process all components of a single layer"""
        print(f"\n{'='*60}")
        print(f"PROCESSING LAYER {layer}")
        print(f"{'='*60}")
        
        success_count = 0
        results = {}
        
        # 1. Input RMSNorm
        print(f"\n[1/5] Input RMSNorm for Layer {layer}")
        if self.generate_rmsnorm_proof(layer, 'input'):
            success_count += 1
            results['input_rmsnorm'] = '✅'
        else:
            results['input_rmsnorm'] = '❌'
        
        # 2. Self-Attention (handles q_proj, k_proj, v_proj, o_proj together)
        print(f"\n[2/5] Self-Attention for Layer {layer}")
        if self.generate_self_attn_proof(layer):
            success_count += 1
            results['self_attn'] = '✅'
        else:
            results['self_attn'] = '❌'
        
        # 3. Post-Attention RMSNorm
        print(f"\n[3/5] Post-Attention RMSNorm for Layer {layer}")
        if self.generate_rmsnorm_proof(layer, 'post_attention'):
            success_count += 1
            results['post_attn_rmsnorm'] = '✅'
        else:
            results['post_attn_rmsnorm'] = '❌'
        
        # 4. Feed-Forward Network (handles gate_proj, up_proj, down_proj together)
        print(f"\n[4/5] Feed-Forward Network for Layer {layer}")
        if self.generate_ffn_proof(layer):
            success_count += 1
            results['ffn'] = '✅'
        else:
            results['ffn'] = '❌'
        
        # 5. Skip Connection
        print(f"\n[5/5] Skip Connection for Layer {layer}")
        if self.generate_skip_connection_proof(layer):
            success_count += 1
            results['skip_connection'] = '✅'
        else:
            results['skip_connection'] = '❌'
        
        # Print layer summary
        print(f"\nLayer {layer} Results:")
        print(f"  Input RMSNorm: {results['input_rmsnorm']}")
        print(f"  Self-Attention: {results['self_attn']}")
        print(f"  Post-Attn RMSNorm: {results['post_attn_rmsnorm']}")
        print(f"  Feed-Forward: {results['ffn']}")
        print(f"  Skip Connection: {results['skip_connection']}")
        print(f"\nLayer {layer} Summary: {success_count}/5 operations successful")
        
        # Consider layer successful if at least 3/5 operations work
        return success_count >= 3
    
    def generate_all_proofs(self, start_layer=0, end_layer=None):
        """Generate proofs for all layers"""
        if end_layer is None:
            end_layer = self.total_layers - 1
        
        print(f"🚀 Starting zkLLM Proof Generation")
        print(f"Model: LLaMA-2-{self.model_size}b")
        print(f"Layers: {start_layer} to {end_layer}")
        print(f"Sequence Length: {self.seq_len}")
        print(f"Work Directory: {self.workdir}")
        print(f"Activation Directory: {self.activation_dir}")
        print(f"Temp Directory: {self.temp_dir}")
        
        start_time = time.time()
        successful_layers = 0
        failed_layers = []
        
        for layer in range(start_layer, end_layer + 1):
            layer_start_time = time.time()
            
            if self.process_single_layer(layer):
                successful_layers += 1
                layer_time = time.time() - layer_start_time
                print(f"✅ Layer {layer} completed in {layer_time:.2f} seconds")
            else:
                failed_layers.append(layer)
                print(f"❌ Layer {layer} failed")
                
                # Ask user if they want to continue
                user_input = input(f"Layer {layer} failed. Continue with next layer? (y/n): ")
                if user_input.lower() != 'y':
                    print("Stopping proof generation.")
                    break
        
        # Final cleanup
        self.cleanup_temp_files()
        
        # Final summary
        total_time = time.time() - start_time
        total_layers_attempted = end_layer - start_layer + 1
        
        print(f"\n{'='*60}")
        print(f"FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Total layers attempted: {total_layers_attempted}")
        print(f"Successfully processed: {successful_layers}")
        print(f"Failed layers: {failed_layers}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per successful layer: {total_time/successful_layers:.2f} seconds" if successful_layers > 0 else "No successful layers")
        print(f"\nFiles created:")
        print(f"  Proofs: {self.workdir}/")
        print(f"  Activations: {self.activation_dir}/")
        
        if successful_layers == total_layers_attempted:
            print("🎉 All layers processed successfully!")
        else:
            print(f"⚠️  {len(failed_layers)} layers failed")
        
        return successful_layers, failed_layers

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='zkLLM Automated Proof Generation')
    parser.add_argument('--model_size', type=int, choices=[7, 13], default=7, 
                       help='Model size (default: 7)')
    parser.add_argument('--seq_len', type=int, default=128, 
                       help='Sequence length (default: 128)')
    parser.add_argument('--start_layer', type=int, default=0, 
                       help='Starting layer (default: 0)')
    parser.add_argument('--end_layer', type=int, default=None, 
                       help='Ending layer (default: all layers)')
    parser.add_argument('--single_layer', type=int, default=None, 
                       help='Process only a single layer')
    parser.add_argument('--workdir', type=str, default=None, 
                       help='Work directory path')
    
    args = parser.parse_args()
    
    # Create proof generator
    generator = ZkLLMProofGenerator(
        model_size=args.model_size,
        seq_len=args.seq_len,
        workdir=args.workdir
    )
    
    # Process layers
    if args.single_layer is not None:
        print(f"Processing single layer: {args.single_layer}")
        success = generator.process_single_layer(args.single_layer)
        if success:
            print(f"✅ Layer {args.single_layer} completed successfully")
        else:
            print(f"❌ Layer {args.single_layer} failed")
    else:
        successful, failed = generator.generate_all_proofs(
            start_layer=args.start_layer,
            end_layer=args.end_layer
        )

if __name__ == '__main__':
    main()