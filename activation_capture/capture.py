"""
Main Activation Capture Orchestrator

Coordinates the entire activation capture pipeline:
- Model loading with 4-bit quantization for 6GB VRAM
- Hook registration
- Inference execution
- Activation serialization
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path
from typing import Optional

from .hooks import ActivationHookManager
from .serializer import ActivationSerializer
from .mapper import ProofPointMapper


class ActivationCaptureManager:
    """
    High-level manager for capturing LLaMA-2 activations.
    
    Orchestrates:
    1. Model loading from HuggingFace cache (with 4-bit quantization)
    2. Hook registration at proof generation points
    3. Inference execution with custom input
    4. Activation serialization to zkLLM format
    """
    
    def __init__(
        self, 
        model_size: int = 7,
        verbose: bool = True,
        use_4bit: bool = True  # Enable 4-bit quantization by default
    ):
        self.model_size = model_size
        # Use model-storage directory in zkllm-ccs2024/
        self.model_cache_dir = "./model-storage"
        self.verbose = verbose
        self.use_4bit = use_4bit
        
        self.model = None
        self.tokenizer = None
        self.hook_manager = None
        self.serializer = None
        
    def load_model(self):
        """Load LLaMA-2 model and tokenizer from local cache"""
        model_card = f"meta-llama/Llama-2-{self.model_size}b-hf"
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Loading LLaMA-2-{self.model_size}B Model")
            print(f"{'='*70}\n")
            print(f"Model: {model_card}")
            print(f"Cache: {self.model_cache_dir}")
        
        # Check GPU availability and memory
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if self.verbose:
                print(f"GPU: {gpu_name}")
                print(f"GPU Memory: {gpu_mem_gb:.2f} GB\n")
        else:
            if self.verbose:
                print(f"⚠ No GPU detected, falling back to CPU\n")
            self.use_4bit = False  # Can't use 4-bit on CPU
        
        # Decide loading strategy
        if self.use_4bit and torch.cuda.is_available():
            self._load_model_4bit(model_card)
        else:
            self._load_model_cpu(model_card)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_card,
            local_files_only=True,
            cache_dir=self.model_cache_dir
        )
        
        # Initialize managers
        self.hook_manager = ActivationHookManager(verbose=self.verbose)
        self.serializer = ActivationSerializer(verbose=self.verbose)
        
        # Print model info
        num_layers = len(self.model.model.layers)
        hidden_size = self.model.config.hidden_size
        
        if self.verbose:
            print(f"  Layers: {num_layers}")
            print(f"  Hidden size: {hidden_size}")
            
            # Check device placement
            first_param = next(self.model.parameters())
            print(f"  Device: {first_param.device}")
            
            # GPU memory usage
            if torch.cuda.is_available() and first_param.is_cuda:
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                print(f"  GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            print()
    
    def _load_model_4bit(self, model_card: str):
        """Load model with 4-bit quantization (fits in 6GB VRAM)"""
        if self.verbose:
            print(f"Loading Strategy: 4-bit Quantization")
            print(f"  • Model will fit in ~4-5GB VRAM")
            print(f"  • Fast inference (~3-5 seconds)")
            print(f"  • Activations quantized to int32 for proofs anyway")
            print(f"  • Installing 'bitsandbytes' if needed...\n")
        
        try:
            # Configure 4-bit quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Load model with 4-bit quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                model_card,
                local_files_only=True,
                cache_dir=self.model_cache_dir,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            if self.verbose:
                print(f"✓ Model loaded successfully (4-bit quantization)")
                
        except ImportError as e:
            if self.verbose:
                print(f"\n⚠ bitsandbytes not installed. Installing now...")
                print(f"  Run: pip install bitsandbytes")
                print(f"  Falling back to CPU loading...\n")
            
            # Fallback to CPU
            self.use_4bit = False
            self._load_model_cpu(model_card)
            
        except Exception as e:
            if self.verbose:
                print(f"\n⚠ 4-bit loading failed: {e}")
                print(f"  Falling back to CPU loading...\n")
            
            # Fallback to CPU
            self.use_4bit = False
            self._load_model_cpu(model_card)
    
    def _load_model_cpu(self, model_card: str):
        """Load model on CPU (uses RAM + swap, slower but works)"""
        if self.verbose:
            print(f"Loading Strategy: CPU-only")
            print(f"  • Uses system RAM (and swap if enabled)")
            print(f"  • Slower inference (~5-10 minutes)")
            print(f"  • Full FP32 precision")
            print(f"  • Enable swap file for better performance\n")
        
        # Load entirely on CPU
        self.model = AutoModelForCausalLM.from_pretrained(
            model_card,
            local_files_only=True,
            cache_dir=self.model_cache_dir,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        if self.verbose:
            print(f"✓ Model loaded successfully on CPU")
            print(f"  ⚠ Inference will be much slower on CPU")
    
    def register_hooks(self, num_layers: Optional[int] = None):
        """
        Register hooks for activation capture.
        
        Args:
            num_layers: Number of layers to capture (None = all layers)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        total_layers = len(self.model.model.layers)
        num_layers = num_layers or total_layers
        
        if self.verbose:
            print(f"{'='*70}")
            print(f"Registering Hooks for {num_layers} Layers")
            print(f"{'='*70}\n")
        
        for layer_idx in range(num_layers):
            layer_module = self.model.model.layers[layer_idx]
            self.hook_manager.register_layer_hooks(layer_idx, layer_module)
        
        total_hooks = len(self.hook_manager.hooks)
        if self.verbose:
            print(f"\n✓ Registered {total_hooks} hooks across {num_layers} layers")
            print(f"  (6 hooks per layer: block-input, input-norm-out, post-attn-residual, ffn-input, mlp-output, block-output)\n")
    
# ... existing code up to capture_from_text ...

    def capture_from_text(
        self, 
        text: str,
        output_dir: str = "temp-files",  # Changed default to match zkLLM
        num_layers: Optional[int] = None
    ) -> dict:
        """
        Capture activations from custom input text.
        
        Args:
            text: Input prompt
            output_dir: Directory to save activations (default: temp-files/)
            num_layers: Number of layers to capture
        
        Returns:
            Results dictionary with metadata
        """
        # Ensure model is loaded
        if self.model is None:
            self.load_model()
        
        # Register hooks
        self.register_hooks(num_layers)
        
        # Tokenize input
        if self.verbose:
            print(f"{'='*70}")
            print(f"Running Inference")
            print(f"{'='*70}\n")
            print(f"Input text: '{text}'")
        
        # Determine device for inputs
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        seq_len = inputs.input_ids.shape[1]
        token_ids = inputs.input_ids[0].tolist()
        
        if self.verbose:
            print(f"Tokens ({seq_len}): {token_ids}")
            decoded_tokens = [self.tokenizer.decode([tid]) for tid in token_ids]
            print(f"Decoded: {decoded_tokens}\n")
        
        # Run inference (activations captured automatically via hooks)
        if self.verbose:
            print("Executing forward pass...")
            if device.type == 'cpu':
                print("⚠ Running on CPU - this may take several minutes...")
        
        import time
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        inference_time = time.time() - start_time
        
        # Get prediction
        next_token_id = torch.argmax(outputs.logits[0, -1, :]).item()
        predicted_token = self.tokenizer.decode([next_token_id])
        
        if self.verbose:
            print(f"✓ Inference complete ({inference_time:.2f} seconds)")
            print(f"  Predicted next token: '{predicted_token}'\n")
        
        # Get captured activations
        activations = self.hook_manager.get_activations()
        
        if self.verbose:
            num_layers_captured = len(activations)
            num_activations = sum(len(layer_acts) for layer_acts in activations.values())
            print(f"Captured activations from {num_layers_captured} layers")
            print(f"Total activation tensors: {num_activations}\n")
            
            # Show sample
            if activations:
                first_layer = list(activations.keys())[0]
                print(f"Sample (Layer {first_layer}):")
                for name, tensor in activations[first_layer].items():
                    print(f"  {name}: {tuple(tensor.shape)}")
                print()
        
        # Save activations
        saved_count = self.serializer.save_batch(activations, output_dir)
        
        # Cleanup
        self.hook_manager.remove_all_hooks()
        self.hook_manager.clear_activations()
        
        result = {
            'input_text': text,
            'predicted_token': predicted_token,
            'seq_len': seq_len,
            'num_activations': saved_count,
            'output_dir': output_dir,
            'inference_time': inference_time
        }
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Capture Complete")
            print(f"{'='*70}")
            print(f"Input: '{text}'")
            print(f"Predicted: '{predicted_token}'")
            print(f"Inference time: {inference_time:.2f}s")
            print(f"Activation files saved: {saved_count}")
            print(f"Output directory: {output_dir}/")
            print(f"{'='*70}\n")
        
        return result