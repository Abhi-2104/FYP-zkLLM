"""
Serialization utilities for saving activations to disk.
Matches zkLLM proof generation expected format EXACTLY.
"""

import torch
import numpy as np
from pathlib import Path
import struct


class ActivationSerializer:
    """
    Handles saving activations in zkLLM-compatible format.
    
    zkLLM proof generation expects files in temp-files/ with specific naming:
    - layer-{i}-input-rmsnorm-input.bin
    - layer-{i}-self-attn-input.bin
    - layer-{i}-post_attention-rmsnorm-input.bin
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.scale_factor = 2 ** 16  # zkLLM uses 2^16 for fixed-point
    
    def save_activation(
        self, 
        tensor: torch.Tensor, 
        filepath: Path
    ):
        """
        Save tensor as int32 binary file (zkLLM format).
        
        Format:
        - No header (raw binary data)
        - int32 values
        - Scaled by 2^16
        - Row-major order (C-style)
        
        Args:
            tensor: PyTorch tensor [seq_len, hidden_size] or [batch, seq_len, hidden_size]
            filepath: Output file path
        """
        # Move to CPU if needed
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        # Squeeze batch dimension if present
        if tensor.dim() == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        
        # Convert to float32, scale, round, convert to int32
        tensor_float = tensor.float().numpy()
        tensor_int = np.round(tensor_float * self.scale_factor).astype(np.int32)
        
        # Save as raw binary (no header, just data)
        tensor_int.tofile(filepath)
        
        if self.verbose:
            print(f"  Saved: {filepath.name}")
            print(f"    Shape: {tensor_int.shape}")
            print(f"    Size: {filepath.stat().st_size / 1024:.2f} KB")
    
    def save_layer_activations(
        self, 
        activations: dict, 
        layer_idx: int,
        output_dir: str = "temp-files"
    ) -> int:
        """
        Save all activations for a single layer with zkLLM-compatible naming.
        
        zkLLM expects these 6 files per layer:
        1. layer-{i}-input-rmsnorm-activation.bin       (input to transformer block)
        2. layer-{i}-self-attn-activation.bin            (output of input RMSNorm, input to Q,K,V)
        3. layer-{i}-post_attention-rmsnorm-activation.bin (after attn skip, input to post-attn norm)
        4. layer-{i}-ffn-activation.bin                  (output of post-attn norm, input to FFN)
        5. layer-{i}-block-input.bin                     (same as #1, for skip connection proof)
        6. layer-{i}-block-output.bin                    (final output after all skips)
        
        Args:
            activations: Dict with keys matching what hooks captured
            layer_idx: Layer index (0-based)
            output_dir: Output directory (default: temp-files)
        
        Returns:
            Number of files saved
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_count = 0
        
        # Mapping: internal hook names → zkLLM expected file names
        # generate_proofs.py expects these exact filenames in activations/ directory
        file_mapping = {
            # Input to the entire transformer block (before input RMSNorm)
            # This matches: layer-{i}-input-rmsnorm-activation.bin
            'block_input': f'layer-{layer_idx}-input-rmsnorm-activation.bin',
            
            # Output of input RMSNorm (input to Q,K,V projections)
            # This matches: layer-{i}-self-attn-activation.bin
            'input_layernorm_output': f'layer-{layer_idx}-self-attn-activation.bin',
            
            # After first skip connection (input to post-attention RMSNorm)
            # This matches: layer-{i}-post_attention-rmsnorm-activation.bin
            'post_attn_residual': f'layer-{layer_idx}-post_attention-rmsnorm-activation.bin',
            
            # Output of post-attention RMSNorm (input to FFN/MLP)
            # This matches: layer-{i}-ffn-activation.bin
            'ffn_input': f'layer-{layer_idx}-ffn-activation.bin',
            
            # Block input for skip connection (duplicate of first one with different name)
            # This matches: layer-{i}-block-input.bin
            # We can reuse 'block_input' but save with different name - handled separately
            
            # Block output for skip connection
            # This matches: layer-{i}-block-output.bin
            'block_output': f'layer-{layer_idx}-block-output.bin',
        }
        
        for internal_name, filename in file_mapping.items():
            if internal_name in activations and activations[internal_name] is not None:
                filepath = output_path / filename
                self.save_activation(activations[internal_name], filepath)
                saved_count += 1
            elif self.verbose:
                print(f"  ⚠ Missing activation: {internal_name} (expected for {filename})")
        
        # Save block-input.bin (duplicate of input-rmsnorm-activation for skip connection)
        if 'block_input' in activations and activations['block_input'] is not None:
            filepath = output_path / f'layer-{layer_idx}-block-input.bin'
            self.save_activation(activations['block_input'], filepath)
            saved_count += 1
        
        return saved_count
    
    def save_batch(
        self, 
        all_activations: dict,
        output_dir: str = "temp-files"
    ) -> int:
        """
        Save activations for multiple layers.
        
        Args:
            all_activations: Dict keyed by layer index, values are activation dicts
            output_dir: Output directory
        
        Returns:
            Total number of files saved
        """
        total_saved = 0
        
        for layer_idx, layer_activations in all_activations.items():
            if self.verbose:
                print(f"\nSaving Layer {layer_idx} activations:")
            
            saved = self.save_layer_activations(
                layer_activations, 
                layer_idx, 
                output_dir
            )
            total_saved += saved
        
        return total_saved