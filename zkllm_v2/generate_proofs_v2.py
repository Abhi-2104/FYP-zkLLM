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
    def __init__(self, model_size=7, device='cpu', start_layer=0, end_layer=31, seq_len=128):
        self.model_size = model_size
        self.device = device
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.seq_len = seq_len
        
        # Setup workdir
        model_name = "Llama-2-7b" if model_size == 7 else "Llama-2-13b"
        self.workdir = Path(f"./zkllm-workdir/{model_name}")
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.activation_dir = Path("./activations")
        self.activation_dir.mkdir(exist_ok=True)
        
        # Track results
        self.results = {}
        
        # Load model once and extract all parameters
        self._load_model_params()
    
    def _load_model_params(self):
        """Load model once and extract parameters needed for proof generation"""
        self.layer_input_eps = {}
        self.layer_post_attn_eps = {}
        
        # 0. Try to load from cache first
        import json
        config_cache = self.workdir / "config.json"
        if config_cache.exists():
            with open(config_cache, 'r') as f:
                cache_data = json.load(f)
                self.embed_dim = cache_data.get('embed_dim')
                self.hidden_dim = cache_data.get('hidden_dim')
                self.num_heads = cache_data.get('num_heads')
                self.variance_epsilon = cache_data.get('variance_epsilon', 1e-5)
                self.layer_input_eps = {int(k): v for k, v in cache_data.get('layer_input_eps', {}).items()}
                self.layer_post_attn_eps = {int(k): v for k, v in cache_data.get('layer_post_attn_eps', {}).items()}
        
        # 1. Check if we actually need to load the model
        needs_loading = False
        target_layers = range(self.start_layer, self.end_layer + 1)
        for i in target_layers:
            w_prefix = f"{self.workdir}/layer-{i}"
            required = [
                f"{w_prefix}-self_attn.q_proj.weight-int.bin",
                f"{w_prefix}-mlp.up_proj.weight-int.bin",
                f"{w_prefix}-input_layernorm.weight-int.bin"
            ]
            if not all(os.path.exists(f) for f in required):
                needs_loading = True
                break
        
        if not needs_loading and hasattr(self, 'embed_dim'):
            print(f"  All required weights for layers {self.start_layer}-{self.end_layer} found. Skipping model load.")
            return

        # 2. Extract and save weights ONLY for requested layers to avoid OOM on 13B
        from transformers import AutoModelForCausalLM
        import fileio_utils
        import gc
        
        # Determine layers to process
        target_layers = range(self.start_layer, self.end_layer + 1)
        # Try to resolve exact snapshot to bypass network
        cache_path = Path("./model-storage") / f"models--meta-llama--Llama-2-{self.model_size}b-hf"
        if cache_path.exists():
            snapshots_dir = cache_path / "snapshots"
            if snapshots_dir.exists() and any(snapshots_dir.iterdir()):
                model_card_path = str(next(snapshots_dir.iterdir()))
            else:
                model_card_path = f"meta-llama/Llama-2-{self.model_size}b-hf"
        else:
            model_card_path = f"meta-llama/Llama-2-{self.model_size}b-hf"

        print(f"\nLoading model {model_card_path} to extract parameters...")
        model = AutoModelForCausalLM.from_pretrained(
            model_card_path, local_files_only=True, cache_dir="./model-storage"
        )
        
        layer0 = model.model.layers[0]
        (self.embed_dim,) = layer0.input_layernorm.weight.shape
        self.variance_epsilon = getattr(model.config, "rms_norm_eps", 1e-6)
        self.hidden_dim = layer0.mlp.up_proj.out_features
        self.num_heads = model.config.num_attention_heads
        
        print(f"  Selective extraction: Saving weights for layers {self.start_layer} to {self.end_layer} to {self.workdir}...")
        
        for i, layer in enumerate(model.model.layers):
            self.layer_input_eps[i] = layer.input_layernorm.variance_epsilon
            self.layer_post_attn_eps[i] = layer.post_attention_layernorm.variance_epsilon
            
            if i in target_layers:
                # Save weights (only if missing to save time)
                w_prefix = f"{self.workdir}/layer-{i}"
                
                # RMSNorm Weights
                if not os.path.exists(f"{w_prefix}-input_layernorm.weight-int.bin"):
                    fileio_utils.save_int(layer.input_layernorm.weight, 1<<16, f"{w_prefix}-input_layernorm.weight-int.bin")
                if not os.path.exists(f"{w_prefix}-post_attention_layernorm.weight-int.bin"):
                    fileio_utils.save_int(layer.post_attention_layernorm.weight, 1<<16, f"{w_prefix}-post_attention_layernorm.weight-int.bin")
                
                # Self-Attention Weights
                if not os.path.exists(f"{w_prefix}-self_attn.q_proj.weight-int.bin"):
                    fileio_utils.save_int(layer.self_attn.q_proj.weight, 1<<16, f"{w_prefix}-self_attn.q_proj.weight-int.bin")
                    fileio_utils.save_int(layer.self_attn.k_proj.weight, 1<<16, f"{w_prefix}-self_attn.k_proj.weight-int.bin")
                    fileio_utils.save_int(layer.self_attn.v_proj.weight, 1<<16, f"{w_prefix}-self_attn.v_proj.weight-int.bin")
                    fileio_utils.save_int(layer.self_attn.o_proj.weight, 1<<16, f"{w_prefix}-self_attn.o_proj.weight-int.bin")
                
                # FFN Weights
                if not os.path.exists(f"{w_prefix}-mlp.gate_proj.weight-int.bin"):
                    fileio_utils.save_int(layer.mlp.gate_proj.weight, 1<<16, f"{w_prefix}-mlp.gate_proj.weight-int.bin")
                if not os.path.exists(f"{w_prefix}-mlp.up_proj.weight-int.bin"):
                    fileio_utils.save_int(layer.mlp.up_proj.weight, 1<<16, f"{w_prefix}-mlp.up_proj.weight-int.bin")
                if not os.path.exists(f"{w_prefix}-mlp.down_proj.weight-int.bin"):
                    fileio_utils.save_int(layer.mlp.down_proj.weight, 1<<16, f"{w_prefix}-mlp.down_proj.weight-int.bin")
                
                # Help GC
                gc.collect()

        # Cache config
        with open(config_cache, 'w') as f:
            json.dump({
                'embed_dim': self.embed_dim,
                'hidden_dim': self.hidden_dim,
                'num_heads': self.num_heads,
                'variance_epsilon': self.variance_epsilon,
                'layer_input_eps': self.layer_input_eps,
                'layer_post_attn_eps': self.layer_post_attn_eps
            }, f)

        del model
        gc.collect()
        
        print(f"  embed_dim={self.embed_dim}, hidden_dim={self.hidden_dim}, heads={self.num_heads}")
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
        
        eps = self.layer_input_eps.get(layer, self.variance_epsilon)
        
        cmd = [
            sys.executable, 'llama-rmsnorm_v2.py',
            str(self.model_size), str(layer), 'input', str(self.seq_len),
            '--input_file', str(input_file),
            '--output_file', str(output_file),
            '--precomputed',
            '--embed_dim', str(self.embed_dim),
            '--variance_epsilon', str(eps)
        ]
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
            sys.executable, 'llama-self-attn_v2.py',
            str(self.model_size), str(layer), str(self.seq_len),
            '--input_file', str(input_file),
            '--output_file', str(output_file),
            '--precomputed',
            '--embed_dim', str(self.embed_dim),
            '--num_heads', str(self.num_heads)
        ]
        
        return self.run_command(cmd, f"Layer {layer} Self-Attention")
    
    def generate_post_attn_rmsnorm(self, layer):
        """
        Generate Post-Attention RMSNorm proof
        
        Input: activations/layer-{N}-self-attn-output.bin
        Output: activations/layer-{N}-ffn-activation.bin
        Proof: zkllm-workdir/.../layer-{N}-post-attn-rmsnorm-proof.bin
        """
        input_file = self.activation_dir / f"layer-{layer}-self-attn-output.bin"
        output_file = self.activation_dir / f"layer-{layer}-ffn-activation.bin"
        
        eps = self.layer_post_attn_eps.get(layer, self.variance_epsilon)
        
        # Use the unified rmsnorm script with which=post_attention
        cmd = [
            sys.executable, 'llama-rmsnorm_v2.py',
            str(self.model_size), str(layer), 'post_attention', str(self.seq_len),
            '--input_file', str(input_file),
            '--output_file', str(output_file),
            '--precomputed',
            '--embed_dim', str(self.embed_dim),
            '--variance_epsilon', str(eps)
        ]
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
            sys.executable, 'llama-ffn_v2.py',
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
            sys.executable, 'llama-skip-connection_v2.py',
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
        if not results['input_rmsnorm']: return False
        
        # 2. Self-Attention
        print(f"\n[2/5] Self-Attention")
        results['self_attn'] = self.generate_self_attention(layer)
        if not results['self_attn']: return False
        
        # 3. Post-Attention RMSNorm
        print(f"\n[3/5] Post-Attention RMSNorm")
        results['post_attn_rmsnorm'] = self.generate_post_attn_rmsnorm(layer)
        if not results['post_attn_rmsnorm']: return False
        
        # 4. Feed-Forward Network
        print(f"\n[4/5] Feed-Forward Network")
        results['ffn'] = self.generate_ffn(layer)
        if not results['ffn']: return False
        
        # 5. Skip Connection
        print(f"\n[5/5] Skip Connection")
        results['skip_connection'] = self.generate_skip_connection(layer)
        if not results['skip_connection']: return False
        
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
        print(f"Model Size: {self.model_size}b")
        print(f"Layers:     {start_layer} to {end_layer}")
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
                # Continue with next layer regardless (standard for non-interactive)
                continue
        
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
  
  # Generate for LLaMA-2-13b instead
  python3 generate_proofs_v2.py --model_size 13 --layer 0
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
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Determine layer range
    if args.layer is not None:
        start_layer = args.layer
        end_layer = args.layer
    else:
        start_layer = args.start_layer
        end_layer = args.end_layer if args.end_layer is not None else start_layer
    
    # Create generator
    generator = ZkLLMProofGeneratorV2(
        model_size=args.model_size,
        device=args.device,
        start_layer=start_layer,
        end_layer=end_layer,
        seq_len=args.seq_len
    )
    
    # Run generation
    successful, failed = generator.generate_all_proofs(
        start_layer=start_layer,
        end_layer=end_layer
    )
    
    # Exit with error code if any failures
    sys.exit(0 if len(failed) == 0 else 1)


if __name__ == '__main__':
    main()