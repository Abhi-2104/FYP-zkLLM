#!/usr/bin/env python3
"""
zkLLM v2 Automated Proof Verification Pipeline

Orchestrates verification of zero-knowledge proofs for all
transformer layer components using v2 verification logic.

Verification order per layer:
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


class ZkLLMProofVerifierV2:
    def __init__(self, model_size=7, seq_len=128, workdir=None):
        self.model_size = model_size
        self.seq_len = seq_len
        self.workdir = workdir or f'./zkllm-workdir/Llama-2-{model_size}b'
        self.total_layers = 32 if model_size == 7 else 40

        self.activation_dir = Path('./activations')

        # Auto-detect real sequence length from capture output if available
        embed_dim = 4096 if model_size == 7 else 5120
        sample_file = self.activation_dir / "layer-0-block-input.bin"
        if sample_file.exists():
            size_bytes = sample_file.stat().st_size
            num_floats = size_bytes // 4
            detected_seq_len = num_floats // embed_dim
            if detected_seq_len > 0:
                self.seq_len = detected_seq_len
                print(f"  Auto-detected seq_len = {self.seq_len} tokens from captured activations.\n")

        # Track results
        self.results = {}

    def compile_verifiers(self):
        """Compile all verification binaries"""
        targets = [
            'verify_rmsnorm_v2',
            'verify_self-attn_v2',
            'verify_ffn_v2',
            'verify_skip-connection_v2',
        ]
        print(f"\nCompiling verification binaries...")
        for target in targets:
            ret = os.system(f'make -f Makefile_v2 {target}')
            if ret != 0:
                print(f"  ❌ Failed to compile {target}")
                return False
            print(f"  ✓ {target}")
        print()
        return True

    def run_command(self, cmd, description=""):
        """Run a command and handle errors"""
        print(f"\n{'─'*60}")
        print(f"[VERIFY] {description}")
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

    def verify_input_rmsnorm(self, layer):
        """
        Verify Input RMSNorm proof

        Proof:  workdir/layer-{N}-input-rmsnorm-proof.bin
        Input:  activations/layer-{N}-block-input.bin
        """
        proof_file = f"{self.workdir}/layer-{layer}-input-rmsnorm-proof.bin"
        input_file = self.activation_dir / f"layer-{layer}-block-input.bin"

        cmd = [
            './verify_rmsnorm_v2',
            proof_file, self.workdir, f'layer-{layer}', 'input',
            str(input_file)
        ]

        return self.run_command(cmd, f"Layer {layer} Input RMSNorm Verification")

    def verify_self_attention(self, layer):
        """
        Verify Self-Attention proof

        Proof:  workdir/layer-{N}-self-attn-proof.bin
        Input:  activations/layer-{N}-input-rmsnorm-activation.bin
        """
        proof_file = f"{self.workdir}/layer-{layer}-self-attn-proof.bin"
        input_file = self.activation_dir / f"layer-{layer}-input-rmsnorm-activation.bin"

        cmd = [
            './verify_self-attn_v2',
            proof_file, self.workdir, f'layer-{layer}',
            str(input_file)
        ]

        return self.run_command(cmd, f"Layer {layer} Self-Attention Verification")

    def verify_post_attn_rmsnorm(self, layer):
        """
        Verify Post-Attention RMSNorm proof

        Proof:  workdir/layer-{N}-post-attn-rmsnorm-proof.bin
        Input:  activations/layer-{N}-self-attn-output.bin
        """
        proof_file = f"{self.workdir}/layer-{layer}-post-attn-rmsnorm-proof.bin"
        input_file = self.activation_dir / f"layer-{layer}-self-attn-output.bin"

        cmd = [
            './verify_rmsnorm_v2',
            proof_file, self.workdir, f'layer-{layer}', 'post_attention',
            str(input_file)
        ]

        return self.run_command(cmd, f"Layer {layer} Post-Attention RMSNorm Verification")

    def verify_ffn(self, layer):
        """
        Verify Feed-Forward Network proof

        Proof:  workdir/layer-{N}-ffn-proof.bin
        Input:  activations/layer-{N}-ffn-activation.bin
        """
        proof_file = f"{self.workdir}/layer-{layer}-ffn-proof.bin"
        input_file = self.activation_dir / f"layer-{layer}-ffn-activation.bin"

        cmd = [
            './verify_ffn_v2',
            proof_file, self.workdir, f'layer-{layer}',
            str(self.seq_len),
            str(input_file)
        ]

        return self.run_command(cmd, f"Layer {layer} FFN Verification")

    def verify_skip_connection(self, layer):
        """
        Verify Skip Connection proof

        Proof:  workdir/layer-{N}-skip-proof.bin  (constructed internally by verifier)
        Input:  activations/layer-{N}-block-input.bin
                activations/layer-{N}-ffn-output.bin
        """
        block_input = self.activation_dir / f"layer-{layer}-block-input.bin"
        ffn_output = self.activation_dir / f"layer-{layer}-ffn-output.bin"

        cmd = [
            './verify_skip-connection_v2',
            self.workdir, f'layer-{layer}',
            str(block_input),
            str(ffn_output)
        ]

        return self.run_command(cmd, f"Layer {layer} Skip Connection Verification")

    def verify_single_layer(self, layer):
        """Verify all components of a single layer"""
        print(f"\n{'='*70}")
        print(f"VERIFYING LAYER {layer}")
        print(f"{'='*70}")

        results = {
            'input_rmsnorm': False,
            'self_attn': False,
            'post_attn_rmsnorm': False,
            'ffn': False,
            'skip_connection': False
        }

        # 1. Input RMSNorm
        print(f"\n[1/5] Input RMSNorm Verification")
        results['input_rmsnorm'] = self.verify_input_rmsnorm(layer)

        # 2. Self-Attention
        print(f"\n[2/5] Self-Attention Verification")
        results['self_attn'] = self.verify_self_attention(layer)

        # 3. Post-Attention RMSNorm
        print(f"\n[3/5] Post-Attention RMSNorm Verification")
        results['post_attn_rmsnorm'] = self.verify_post_attn_rmsnorm(layer)

        # 4. Feed-Forward Network
        print(f"\n[4/5] Feed-Forward Network Verification")
        results['ffn'] = self.verify_ffn(layer)

        # 5. Skip Connection
        print(f"\n[5/5] Skip Connection Verification")
        results['skip_connection'] = self.verify_skip_connection(layer)

        # Print layer summary
        success_count = sum(results.values())
        print(f"\n{'─'*70}")
        print(f"LAYER {layer} VERIFICATION SUMMARY")
        print(f"{'─'*70}")
        print(f"  Input RMSNorm:      {'✅' if results['input_rmsnorm'] else '❌'}")
        print(f"  Self-Attention:     {'✅' if results['self_attn'] else '❌'}")
        print(f"  Post-Attn RMSNorm:  {'✅' if results['post_attn_rmsnorm'] else '❌'}")
        print(f"  Feed-Forward:       {'✅' if results['ffn'] else '❌'}")
        print(f"  Skip Connection:    {'✅' if results['skip_connection'] else '❌'}")
        print(f"{'─'*70}")
        print(f"  Result: {success_count}/5 verifications passed")

        self.results[layer] = results
        return success_count == 5

    def verify_all_proofs(self, start_layer=0, end_layer=None):
        """Verify proofs for specified layers"""
        if end_layer is None:
            end_layer = start_layer

        print(f"\n{'='*70}")
        print(f"zkLLM v2 PROOF VERIFICATION PIPELINE")
        print(f"{'='*70}")
        print(f"Model Size: {self.model_size}b")
        print(f"Layers:     {start_layer} to {end_layer}")
        print(f"Seq Length: {self.seq_len}")
        print(f"Work Dir:   {self.workdir}")
        print(f"Activations:{self.activation_dir}")
        print(f"{'='*70}")

        # Compile all verifiers once
        if not self.compile_verifiers():
            print("❌ Compilation failed. Aborting.")
            return 0, list(range(start_layer, end_layer + 1))

        start_time = time.time()
        successful_layers = 0
        failed_layers = []

        for layer in range(start_layer, end_layer + 1):
            layer_start = time.time()

            if self.verify_single_layer(layer):
                successful_layers += 1
                layer_time = time.time() - layer_start
                print(f"\n✅ Layer {layer} fully verified in {layer_time:.1f}s")
            else:
                failed_layers.append(layer)
                print(f"\n❌ Layer {layer} had verification failures")

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
        print(f"FINAL VERIFICATION SUMMARY")
        print(f"{'='*70}")
        print(f"Layers attempted:     {end_layer - start_layer + 1}")
        print(f"Fully verified:       {successful_layers}")
        print(f"With failures:        {len(failed_layers)}")
        print(f"Total time:           {total_time:.1f}s")

        if failed_layers:
            print(f"Failed layers:        {failed_layers}")

        print(f"\nProofs verified from: {self.workdir}/")
        print(f"Activations at:       {self.activation_dir}/")
        print(f"{'='*70}")

        return successful_layers, failed_layers


def main():
    parser = argparse.ArgumentParser(
        description='zkLLM v2 Automated Proof Verification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify proofs for layer 0 (LLaMA-2-7b, default)
  python3 verify_proofs_v2.py --layer 0

  # Verify proofs for layers 0 through 5
  python3 verify_proofs_v2.py --start_layer 0 --end_layer 5

  # Verify for LLaMA-2-13b
  python3 verify_proofs_v2.py --model_size 13 --layer 0

  # Custom sequence length
  python3 verify_proofs_v2.py --layer 0 --seq_len 64
        """
    )
    parser.add_argument('--model_size', type=int, choices=[7, 13], default=7,
                        help='Model size (default: 7)')
    parser.add_argument('--seq_len', type=int, default=128,
                        help='Sequence length (default: 128)')
    parser.add_argument('--layer', type=int, default=None,
                        help='Verify a single layer')
    parser.add_argument('--start_layer', type=int, default=0,
                        help='Starting layer (default: 0)')
    parser.add_argument('--end_layer', type=int, default=None,
                        help='Ending layer (default: same as start_layer)')
    parser.add_argument('--workdir', type=str, default=None,
                        help='Work directory path')

    args = parser.parse_args()

    # Create verifier
    verifier = ZkLLMProofVerifierV2(
        model_size=args.model_size,
        seq_len=args.seq_len,
        workdir=args.workdir
    )

    # Determine layer range
    if args.layer is not None:
        start_layer = args.layer
        end_layer = args.layer
    else:
        start_layer = args.start_layer
        end_layer = args.end_layer if args.end_layer is not None else start_layer

    # Run verification
    successful, failed = verifier.verify_all_proofs(
        start_layer=start_layer,
        end_layer=end_layer
    )

    # Exit with error code if any failures
    sys.exit(0 if len(failed) == 0 else 1)


if __name__ == '__main__':
    main()
