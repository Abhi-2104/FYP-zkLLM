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
import shutil
from pathlib import Path


class ZkLLMProofGeneratorV2:
    def __init__(
        self,
        model_size=7,
        device='cpu',
        start_layer=0,
        end_layer=31,
        seq_len=128,
        act_dir=None,
        workdir=None,
        run_id=None,
        model_card=None
    ):
        self.model_size = model_size
        self.device = device
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.seq_len = seq_len
        self.model_card = model_card
        
        # Setup workdir
        model_name = "Llama-2-7b" if model_size == 7 else "Llama-2-13b"
        self.base_workdir = Path(workdir) if workdir else Path(f"./zkllm-workdir/{model_name}")
        self.run_id = str(run_id) if run_id is not None else None
        self.workdir = self.base_workdir / self.run_id if self.run_id else self.base_workdir
        self.workdir.mkdir(parents=True, exist_ok=True)
        if act_dir:
            self.activation_dir = Path(act_dir)
        elif self.run_id:
            self.activation_dir = Path("./activations") / self.run_id
        else:
            self.activation_dir = Path("./activations")
        self.activation_dir.mkdir(parents=True, exist_ok=True)
        
        # Track results
        self.results = {}
        
        # Ensure base artifacts are visible in the session workdir
        self._ensure_common_links()
        self._ensure_required_weight_links()
        
        # Load model once and extract all parameters
        self._load_model_params()
    
    def _link_from_base(self, relative_path):
        if self.base_workdir == self.workdir:
            return
        src = self.base_workdir / relative_path
        dest = self.workdir / relative_path
        if dest.exists() or not src.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(src, dest)
        except OSError:
            shutil.copy2(src, dest)

    def _ensure_common_links(self):
        self._link_from_base("config.json")
        common_files = [
            "input_layernorm.weight-pp.bin",
            "post_attention_layernorm.weight-pp.bin",
            "self_attn.q_proj.weight-pp.bin",
            "self_attn.k_proj.weight-pp.bin",
            "self_attn.v_proj.weight-pp.bin",
            "self_attn.o_proj.weight-pp.bin",
            "mlp.gate_proj.weight-pp.bin",
            "mlp.up_proj.weight-pp.bin",
            "mlp.down_proj.weight-pp.bin",
        ]
        for fname in common_files:
            self._link_from_base(fname)

    def _ensure_layer_links(self, layer):
        self._ensure_common_links()
        layer_prefix = f"layer-{layer}"
        names = [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ]
        for name in names:
            self._link_from_base(f"{layer_prefix}-{name}-int.bin")
            self._link_from_base(f"{layer_prefix}-{name}-commitment.bin")

    def _ensure_required_weight_links(self):
        for i in range(self.start_layer, self.end_layer + 1):
            self._link_from_base(f"layer-{i}-self_attn.q_proj.weight-int.bin")
            self._link_from_base(f"layer-{i}-mlp.up_proj.weight-int.bin")
            self._link_from_base(f"layer-{i}-input_layernorm.weight-int.bin")

    def _truncate_activation_files(self, layer):
        """Truncate raw activation bin files to seq_len * embed_dim int32 elements.
        
        Activation capture may produce files with more tokens than the configured
        seq_len (e.g., capture for 10 decode steps but prove for 1 token).
        All CUDA binaries expect exactly seq_len * embed_dim int32 elements.
        """
        import numpy as np
        expected_bytes = self.seq_len * self.embed_dim * 4  # int32 = 4 bytes
        
        # Files that come from the raw activation capture and may have extra tokens
        raw_files = [
            self.activation_dir / f"layer-{layer}-block-input.bin",
            self.activation_dir / f"layer-{layer}-self-attn-output.bin",
        ]
        
        for fpath in raw_files:
            if fpath.exists() and fpath.stat().st_size > expected_bytes:
                data = np.fromfile(str(fpath), dtype=np.int32, count=self.seq_len * self.embed_dim)
                data.tofile(str(fpath))
                print(f"  ✂️  Truncated {fpath.name} to {self.seq_len}×{self.embed_dim} ({len(data)} int32s)")

    def _load_model_params(self):
        """Load model once and extract parameters needed for proof generation"""
        self.layer_input_eps = {}
        self.layer_post_attn_eps = {}
        
        # 0. Try to load config cache first
        import json
        config_cache = self.workdir / "config.json"
        cache_loaded = False
        if config_cache.exists():
            with open(config_cache, 'r') as f:
                cache_data = json.load(f)
                self.embed_dim = cache_data.get('embed_dim')
                self.hidden_dim = cache_data.get('hidden_dim')
                self.num_heads = cache_data.get('num_heads')
                self.variance_epsilon = cache_data.get('variance_epsilon', 1e-5)
                self.layer_input_eps = {int(k): v for k, v in cache_data.get('layer_input_eps', {}).items()}
                self.layer_post_attn_eps = {int(k): v for k, v in cache_data.get('layer_post_attn_eps', {}).items()}
                cache_loaded = all([
                    self.embed_dim is not None,
                    self.hidden_dim is not None,
                    self.num_heads is not None,
                ])

        # Auto-detect real sequence length from capture output if available
        sample_file = self.activation_dir / "layer-0-block-input.bin"
        embed_dim = getattr(self, "embed_dim", None)
        if embed_dim is None:
            embed_dim = 4096 if self.model_size == 7 else 5120
        if sample_file.exists() and embed_dim:
            size_bytes = sample_file.stat().st_size
            num_floats = size_bytes // 4
            detected_seq_len = num_floats // embed_dim
            if detected_seq_len > 0:
                if detected_seq_len < self.seq_len:
                    self.seq_len = detected_seq_len
                    print(f"  Auto-adjusted seq_len down to {self.seq_len} tokens based on captured activations.\n")
                elif detected_seq_len > self.seq_len:
                    print(
                        f"  Captured activations have {detected_seq_len} tokens; "
                        f"keeping configured proof seq_len={self.seq_len}."
                    )
        
        # 1. Load model metadata (old v2 behavior) and avoid rewriting layer weights.
        from transformers import AutoModelForCausalLM
        import gc
        
        # Try to resolve exact snapshot to bypass network
        model_card_path = self.model_card
        if not model_card_path:
            cache_path = Path("./model-storage") / f"models--meta-llama--Llama-2-{self.model_size}b-hf"
            if cache_path.exists():
                snapshots_dir = cache_path / "snapshots"
                if snapshots_dir.exists() and any(snapshots_dir.iterdir()):
                    model_card_path = str(next(snapshots_dir.iterdir()))
                else:
                    model_card_path = f"meta-llama/Llama-2-{self.model_size}b-hf"
            else:
                model_card_path = f"meta-llama/Llama-2-{self.model_size}b-hf"

        model = None
        model_loaded = False
        try:
            print(f"\nLoading model {model_card_path} for metadata...")
            model = AutoModelForCausalLM.from_pretrained(
                model_card_path, local_files_only=True, cache_dir="./model-storage"
            )
            model_loaded = True

            layer0 = model.model.layers[0]
            (self.embed_dim,) = layer0.input_layernorm.weight.shape
            self.variance_epsilon = getattr(model.config, "rms_norm_eps", 1e-6)
            self.hidden_dim = layer0.mlp.up_proj.out_features
            self.num_heads = model.config.num_attention_heads

            for i, layer in enumerate(model.model.layers):
                self.layer_input_eps[i] = layer.input_layernorm.variance_epsilon
                self.layer_post_attn_eps[i] = layer.post_attention_layernorm.variance_epsilon
        except Exception as e:
            if not cache_loaded:
                raise RuntimeError(f"Failed to load model metadata and no cache was available: {e}")
            print(f"  Warning: failed to load model metadata, using cached config instead ({e})")

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

        if model is not None:
            del model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        
        source = "model" if model_loaded else "cache"
        print(f"  embed_dim={self.embed_dim}, hidden_dim={self.hidden_dim}, heads={self.num_heads} (from {source})")
        print(f"  Model metadata phase completed.\n")
        
        # Sequence length already auto-detected above if activations are present
    
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
        finally:
            self._python_gpu_cleanup()

    def _python_gpu_cleanup(self):
        """Best-effort cleanup between component runs to keep memory pressure low."""
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
    
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
            '--workdir', str(self.workdir),
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
            '--workdir', str(self.workdir),
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
            '--workdir', str(self.workdir),
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
            '--workdir', str(self.workdir),
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
            '--workdir', str(self.workdir),
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
        self._ensure_layer_links(layer)
        self._truncate_activation_files(layer)
        
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
        print(f"Base Work:  {self.base_workdir}")
        print(f"Proof Dir:  {self.workdir}")
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
        print(f"Base weights at:   {self.base_workdir}/")
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
    parser.add_argument('--act_dir', type=str, default=None,
                        help='Activations directory')
    parser.add_argument('--workdir', type=str, default=None,
                        help='Base workdir for model artifacts')
    parser.add_argument('--run_id', type=str, default=None,
                        help='Session/run ID (proofs saved under base workdir/<run_id>)')
    parser.add_argument('--model_card', type=str, default=None,
                        help='Override model card or snapshot path')
    
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
        seq_len=args.seq_len,
        act_dir=args.act_dir,
        workdir=args.workdir,
        run_id=args.run_id,
        model_card=args.model_card
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