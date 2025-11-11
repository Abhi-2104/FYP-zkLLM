"""
LLaMA Layer to Proof Point Mapper

Maps LLaMA-2 transformer layers to zkLLM proof generation points,
defining which activations need to be captured for each component.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class ProofPoint:
    """Represents a computation point where proof is generated"""
    activation_name: str      # e.g., 'input', 'attn-output'
    proof_component: str      # e.g., 'rmsnorm', 'self-attn', 'ffn'
    description: str          # Human-readable description


class ProofPointMapper:
    """
    Maps LLaMA-2 architecture to zkLLM proof generation points.
    
    For each transformer layer, identifies the exact points where
    activations must be captured to generate proofs for:
    - Input RMSNorm
    - Self-Attention
    - Post-Attention RMSNorm
    - FFN
    - Skip Connections
    """
    
    # Define proof points for each layer
    LAYER_PROOF_POINTS = [
        ProofPoint(
            activation_name='input',
            proof_component='rmsnorm-input',
            description='Input to layer (before input RMSNorm)'
        ),
        ProofPoint(
            activation_name='attn-output',
            proof_component='self-attn',
            description='Self-attention output (before post-attn RMSNorm)'
        ),
        ProofPoint(
            activation_name='post-norm',
            proof_component='rmsnorm-post-attn',
            description='Post-attention RMSNorm output (input to FFN)'
        ),
        ProofPoint(
            activation_name='ffn-output',
            proof_component='ffn',
            description='FFN output (for skip connection)'
        ),
    ]
    
    @classmethod
    def get_activation_name(cls, layer_idx: int, point_name: str) -> str:
        """
        Generate standardized activation name.
        
        Args:
            layer_idx: Transformer layer index
            point_name: Proof point name (e.g., 'input', 'attn-output')
        
        Returns:
            Full activation name (e.g., 'layer-0-input')
        """
        return f"layer-{layer_idx}-{point_name}"
    
    @classmethod
    def get_proof_points_for_layer(cls, layer_idx: int) -> List[ProofPoint]:
        """Get all proof points for a specific layer"""
        return cls.LAYER_PROOF_POINTS
    
    @classmethod
    def map_to_proof_component(cls, activation_name: str) -> str:
        """
        Map activation name to proof component.
        
        Examples:
            'layer-0-input' → 'rmsnorm-input'
            'layer-0-attn-output' → 'self-attn'
        """
        for point in cls.LAYER_PROOF_POINTS:
            if point.activation_name in activation_name:
                return point.proof_component
        
        return 'unknown'
    
    @classmethod
    def print_mapping(cls, num_layers: int = 1):
        """Print architecture mapping for documentation"""
        print("\n" + "=" * 70)
        print("LLaMA-2 Layer → zkLLM Proof Point Mapping")
        print("=" * 70 + "\n")
        
        for layer_idx in range(num_layers):
            print(f"Layer {layer_idx}:")
            for point in cls.LAYER_PROOF_POINTS:
                full_name = cls.get_activation_name(layer_idx, point.activation_name)
                print(f"  {full_name:30s} → {point.proof_component:20s} | {point.description}")
            print()