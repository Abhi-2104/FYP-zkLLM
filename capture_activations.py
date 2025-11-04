#!/usr/bin/env python3
# filepath: /home/abhi/DISK2/FYP/zkllm-ccs2024/capture_activations.py
"""
CLI tool for capturing LLaMA-2 activations for zkLLM proof generation.

Usage:
    python capture_activations.py --text "Hello world"
    python capture_activations.py --text "Hello world" --num_layers 8
"""

import argparse
import sys
from pathlib import Path

from activation_capture import ActivationCaptureManager, ProofPointMapper


def check_dependencies():
    """Check if required packages are installed"""
    missing = []
    
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    try:
        import accelerate
    except ImportError:
        missing.append("accelerate")
    
    if missing:
        print(f"\n⚠ Missing required packages: {', '.join(missing)}")
        print(f"\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        print()
        sys.exit(1)
    
    # Check for bitsandbytes (optional but recommended)
    try:
        import bitsandbytes
    except ImportError:
        print(f"\n⚠ Optional package 'bitsandbytes' not installed")
        print(f"  For 4-bit quantization (faster, fits in 6GB VRAM):")
        print(f"  pip install bitsandbytes")
        print(f"\n  Without it, model will load on CPU (slower but works)\n")


def main():
    parser = argparse.ArgumentParser(
        description='Capture LLaMA-2 activations for zkLLM proof generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Capture ALL layers (default for 7B model = 32 layers)
  python capture_activations.py --text "The capital of France is"
  
  # Capture specific number of layers (e.g., first 3 layers)
  python capture_activations.py --text "The capital of France is" --num_layers 3
  
  # Capture first 8 layers
  python capture_activations.py --text "Hello world" --num_layers 8
  
  # Use 13B model (40 layers total by default)
  python capture_activations.py --text "Test" --model_size 13
  
  # Custom output directory (inside zkllm-ccs2024/)
  python capture_activations.py --text "Test" --output_dir my_activations
  
  # Force CPU loading (slower but uses swap file)
  python capture_activations.py --text "Test" --cpu
  
  # Show architecture mapping
  python capture_activations.py --show_mapping
  
Requirements:
  - transformers
  - torch
  - accelerate
  - bitsandbytes (optional, for 4-bit quantization)
  
  Install with: pip install transformers torch accelerate bitsandbytes
        """
    )
    
    parser.add_argument('--text', type=str,
                        help='Input text prompt for inference')
    parser.add_argument('--model_size', type=int, choices=[7, 13], default=7,
                        help='LLaMA-2 model size (7B or 13B, default: 7)')
    parser.add_argument('--num_layers', type=int, default=None,
                        help='Number of layers to capture (default: ALL layers in model)')
    parser.add_argument('--output_dir', type=str, default='activations',
                        help='Output directory for saved activations (default: activations/)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU loading (slower but uses swap file)')
    parser.add_argument('--show_mapping', action='store_true',
                        help='Show architecture mapping and exit')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')
    parser.add_argument('--check_deps', action='store_true',
                        help='Check dependencies and exit')
    
    args = parser.parse_args()
    
    # Check dependencies if requested
    if args.check_deps:
        check_dependencies()
        print("✓ All required dependencies installed\n")
        return
    
    # Show mapping if requested
    if args.show_mapping:
        ProofPointMapper.print_mapping(num_layers=2)
        return
    
    # Require text input if not showing mapping
    if not args.text:
        parser.error("--text is required (unless using --show_mapping or --check_deps)")
    
    # Check dependencies
    check_dependencies()
    
    # Determine number of layers
    # LLaMA-2-7B has 32 layers, LLaMA-2-13B has 40 layers
    total_layers = 32 if args.model_size == 7 else 40
    num_layers = args.num_layers if args.num_layers is not None else total_layers
    
    # Validate num_layers
    if num_layers > total_layers:
        print(f"\n⚠ WARNING: Requested {num_layers} layers, but LLaMA-2-{args.model_size}B only has {total_layers} layers.")
        print(f"  Will capture all {total_layers} layers instead.\n")
        num_layers = total_layers
    
    # Print capture info
    if not args.quiet:
        print(f"\n{'='*70}")
        print(f"Activation Capture Configuration")
        print(f"{'='*70}")
        print(f"Model: LLaMA-2-{args.model_size}B")
        print(f"Total layers in model: {total_layers}")
        print(f"Layers to capture: {num_layers} ({'ALL' if num_layers == total_layers else 'PARTIAL'})")
        print(f"Activations per layer: 4 (input, attn-output, post-norm, ffn-output)")
        print(f"Total activations to capture: {num_layers * 4}")
        print(f"Output directory: {args.output_dir}/")
        print(f"Loading mode: {'CPU-only' if args.cpu else '4-bit GPU (if available)'}")
        print(f"{'='*70}\n")
    
    # Create capture manager
    manager = ActivationCaptureManager(
        model_size=args.model_size,
        verbose=not args.quiet,
        use_4bit=not args.cpu  # Disable 4-bit if --cpu flag is set
    )
    
    # Run capture
    try:
        result = manager.capture_from_text(
            text=args.text,
            output_dir=args.output_dir,
            num_layers=num_layers
        )
        
        print(f"\n{'='*70}")
        print(f"✓ CAPTURE COMPLETE")
        print(f"{'='*70}")
        print(f"Input text: '{result['input_text']}'")
        print(f"Predicted next token: '{result['predicted_token']}'")
        print(f"Sequence length: {result['seq_len']} tokens")
        print(f"Inference time: {result['inference_time']:.2f} seconds")
        print(f"Layers captured: {num_layers}/{total_layers}")
        print(f"Activations saved: {result['num_activations']}")
        print(f"Output directory: {result['output_dir']}/")
        print(f"{'='*70}")
        print(f"\n✓ Ready for proof generation!")
        print(f"\n{'='*70}")
        print(f"⚠️  IMPORTANT: Use this command for proof generation:")
        print(f"{'='*70}")
        print(f"python generate_proofs.py --seq_len {result['seq_len']} --single_layer <layer_num>")
        print(f"# OR for multiple layers:")
        print(f"python generate_proofs.py --seq_len {result['seq_len']} --start_layer 0 --end_layer {num_layers-1}")
        print(f"{'='*70}\n")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠ Interrupted by user\n")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"✗ CAPTURE FAILED")
        print(f"{'='*70}")
        print(f"Error: {e}\n")
        
        # Provide helpful hints based on error type
        error_str = str(e).lower()
        
        if "out of memory" in error_str or "oom" in error_str:
            print("Hint: GPU out of memory")
            print("  Try one of these solutions:")
            print("  1. Install bitsandbytes for 4-bit quantization:")
            print("     pip install bitsandbytes")
            print("  2. Use CPU mode (slower but works with swap):")
            print("     python capture_activations.py --text '...' --cpu")
            print("  3. Reduce number of layers:")
            print("     python capture_activations.py --text '...' --num_layers 8")
            print()
            
        elif "accelerate" in error_str:
            print("Hint: Missing accelerate package")
            print("  Install with: pip install 'accelerate>=0.26.0'")
            print()
            
        elif "bitsandbytes" in error_str:
            print("Hint: 4-bit quantization requires bitsandbytes")
            print("  Install with: pip install bitsandbytes")
            print("  Or use CPU mode: --cpu")
            print()
        
        import traceback
        traceback.print_exc()
        print(f"{'='*70}\n")
        sys.exit(1)


if __name__ == '__main__':
    main()