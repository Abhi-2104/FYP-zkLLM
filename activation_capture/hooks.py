"""
Hook Management System for LLaMA-2 Activation Capture

Captures activations at EXACT points needed by zkLLM proof generation.
"""

import torch
import torch.nn as nn
from typing import Dict, List
from collections import OrderedDict


class ActivationHookManager:
    """
    Manages registration and lifecycle of forward hooks for activation capture.
    
    Captures at 6 critical points per layer (matching zkLLM proof pipeline):
    1. Block input (before input RMSNorm) → layer-{i}-input-rmsnorm-activation.bin
    2. Input layernorm output (after input RMSNorm) → layer-{i}-self-attn-activation.bin
    3. Post-attention residual (after 1st skip) → layer-{i}-post_attention-rmsnorm-activation.bin
    4. Post-attention layernorm output (input to FFN) → layer-{i}-ffn-activation.bin
    5. Block input (for skip connection) → layer-{i}-block-input.bin
    6. Block output (after final skip) → layer-{i}-block-output.bin
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        # Changed to dict of dicts: {layer_idx: {activation_name: tensor}}
        self.activations: Dict[int, Dict[str, torch.Tensor]] = {}
        
    def register_layer_hooks(self, layer_idx: int, layer_module: nn.Module):
        """
        Register hooks for a single transformer layer.
        
        Captures 3 activations needed by zkLLM:
        1. block_input - Input to entire layer (before input RMSNorm)
        2. input_layernorm_output - Output of input RMSNorm (input to Q,K,V)
        3. post_attn_residual - After attention + skip (input to post-attn RMSNorm)
        
        Args:
            layer_idx: Index of the transformer layer
            layer_module: The actual nn.Module for this layer
        """
        
        # Initialize storage for this layer
        if layer_idx not in self.activations:
            self.activations[layer_idx] = {}
        
        # HOOK 1: Capture block input (BEFORE input RMSNorm)
        # This is: layer-{i}-input-rmsnorm-input.bin
        def capture_block_input(module, input):
            """Pre-hook: captures input to the entire transformer block"""
            hidden_states = input[0]  # Extract from tuple
            self.activations[layer_idx]['block_input'] = hidden_states.detach().clone()
            
            if self.verbose:
                print(f"  ✓ Captured layer-{layer_idx}-block-input: {tuple(hidden_states.shape)}")
        
        handle = layer_module.register_forward_pre_hook(capture_block_input)
        self.hooks.append(handle)
        
        # HOOK 2: Capture input layernorm output (AFTER input RMSNorm)
        # This is: layer-{i}-self-attn-input.bin
        # This becomes input to Q, K, V projections
        def capture_input_norm_output(module, input, output):
            """Forward hook: captures output of input RMSNorm"""
            self.activations[layer_idx]['input_layernorm_output'] = output.detach().clone()
            
            if self.verbose:
                print(f"  ✓ Captured layer-{layer_idx}-input-norm-output: {tuple(output.shape)}")
        
        handle = layer_module.input_layernorm.register_forward_hook(capture_input_norm_output)
        self.hooks.append(handle)
        
        # HOOK 3: Capture post-attention residual (AFTER first skip connection)
        # This is: layer-{i}-post_attention-rmsnorm-input.bin
        # We need to capture the state AFTER: residual + attn_output
        # This happens BEFORE post_attention_layernorm
        def capture_post_attn_residual(module, input):
            """Pre-hook on post_attention_layernorm: captures input to it"""
            hidden_states = input[0]  # This is residual + attn_output
            self.activations[layer_idx]['post_attn_residual'] = hidden_states.detach().clone()
            
            if self.verbose:
                print(f"  ✓ Captured layer-{layer_idx}-post-attn-residual: {tuple(hidden_states.shape)}")
        
        handle = layer_module.post_attention_layernorm.register_forward_pre_hook(capture_post_attn_residual)
        self.hooks.append(handle)
        
        # HOOK 4: Capture post-attention layernorm output (input to FFN/MLP)
        # This is: layer-{i}-ffn-activation.bin
        def capture_ffn_input(module, input, output):
            """Forward hook: captures output of post_attention_layernorm (input to FFN)"""
            self.activations[layer_idx]['ffn_input'] = output.detach().clone()
            
            if self.verbose:
                print(f"  ✓ Captured layer-{layer_idx}-ffn-input: {tuple(output.shape)}")
        
        handle = layer_module.post_attention_layernorm.register_forward_hook(capture_ffn_input)
        self.hooks.append(handle)
        
        # HOOK 5: Capture MLP output (for final skip connection)
        # This will be used to compute block_output = post_attn_residual + mlp_output
        def capture_mlp_output(module, input, output):
            """Forward hook: captures output of MLP/FFN"""
            self.activations[layer_idx]['mlp_output'] = output.detach().clone()
            
            if self.verbose:
                print(f"  ✓ Captured layer-{layer_idx}-mlp-output: {tuple(output.shape)}")
        
        handle = layer_module.mlp.register_forward_hook(capture_mlp_output)
        self.hooks.append(handle)
        
        # HOOK 6: Capture block output (AFTER final skip connection)
        # This is: layer-{i}-block-output.bin
        # Block output = post_attn_residual + mlp_output
        # We'll compute this in post-processing since there's no explicit skip module
        # The block output is the final output of the layer module
        def capture_block_output(module, input, output):
            """Forward hook: captures final output of transformer block"""
            # output is a tuple (hidden_states, ...) for some models
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            self.activations[layer_idx]['block_output'] = hidden_states.detach().clone()
            
            if self.verbose:
                print(f"  ✓ Captured layer-{layer_idx}-block-output: {tuple(hidden_states.shape)}")
        
        handle = layer_module.register_forward_hook(capture_block_output)
        self.hooks.append(handle)
        
        if self.verbose:
            print(f"  → Registered 6 hooks for layer {layer_idx}")
    
    def get_activations(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Return all captured activations.
        
        Returns:
            Dict mapping layer_idx to dict of activations:
            {
                0: {
                    'block_input': tensor,
                    'input_layernorm_output': tensor,
                    'post_attn_residual': tensor
                },
                1: { ... },
                ...
            }
        """
        return self.activations
    
    def clear_activations(self):
        """Clear stored activations to free memory"""
        self.activations.clear()
        torch.cuda.empty_cache()
    
    def remove_all_hooks(self):
        """Remove all registered hooks"""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        if self.verbose:
            print(f"\n✓ Removed all hooks")
    
    def __del__(self):
        """Ensure cleanup on deletion"""
        self.remove_all_hooks()