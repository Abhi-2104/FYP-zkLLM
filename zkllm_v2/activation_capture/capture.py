"""
Main Activation Capture Orchestrator

Coordinates the entire activation capture pipeline:
- Model loading with 4-bit quantization for 6GB VRAM
- Hook registration
- Inference execution
- Activation serialization
"""

import torch
import gc
import random
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
        # Use relative path to model-storage directory as requested
        self.model_cache_dir = "./model-storage"
        self.verbose = verbose
        self.use_4bit = use_4bit
        
        self.model = None
        self.tokenizer = None
        self.hook_manager = None
        self.serializer = None
        
    def load_model(self):
        """Load LLaMA-2 model and tokenizer from local cache"""
        model_name = f"meta-llama/Llama-2-{self.model_size}b-hf"
        
        # Try to resolve to the exact snapshot path to avoid network calls
        cache_path = Path(self.model_cache_dir) / f"models--meta-llama--Llama-2-{self.model_size}b-hf"
        if cache_path.exists():
            snapshots_dir = cache_path / "snapshots"
            if snapshots_dir.exists() and any(snapshots_dir.iterdir()):
                model_card = str(next(snapshots_dir.iterdir()))
            else:
                model_card = model_name
        else:
            model_card = model_name
            
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Loading LLaMA-2-{self.model_size}B Model")
            print(f"{'='*70}\n")
            print(f"Model ID: {model_name}")
            print(f"Path: {model_card}")
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
        # Use float16 to save 50% memory (26GB for 13B instead of 52GB)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_card,
            local_files_only=True,
            cache_dir=self.model_cache_dir,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        if self.verbose:
            print(f"✓ Model loaded successfully on CPU")
            print(f"  ⚠ Inference will be much slower on CPU")
    
    def register_hooks(self, num_layers: Optional[int] = None, target_step_idx: int = 0, target_token_idx: Optional[int] = -1):
        """
        Register hooks for activation capture.
        
        Args:
            num_layers: Number of layers to capture (None = all layers)
            target_step_idx: Which generation step to capture (0 = prefill)
            target_token_idx: Which token within that step to capture (-1 = last)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        total_layers = len(self.model.model.layers)
        num_layers = num_layers or total_layers
        
        if self.verbose:
            print(f"{'='*70}")
            print(f"Registering Hooks for {num_layers} Layers")
            print(f"  Target Step: {target_step_idx}, Target Token: {target_token_idx}")
            print(f"{'='*70}\n")
        
        # Configure hook manager
        self.hook_manager.target_step_idx = target_step_idx
        self.hook_manager.current_step = 0
        self.hook_manager.target_token_idx = target_token_idx
        self.hook_manager.activations.clear()
        
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
        num_layers: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        max_new_tokens: int = 1,
        capture_mode: str = "auto",
        target_generated_step: Optional[int] = None,
    ) -> dict:
        """
        Capture activations from custom input text.
        
        Args:
            text: Input prompt
            output_dir: Directory to save activations (default: temp-files/)
            num_layers: Number of layers to capture
            max_seq_len: Optional max token length for truncation
            max_new_tokens: Number of continuation tokens to greedily decode
            capture_mode: One of:
                - "auto": prefill for single-token requests, random generated-step for multi-token requests
                - "prefill": capture prompt prefill forward pass (legacy behavior)
                - "random_generated": capture at a random decode step k in [1, max_new_tokens]
                - "specific_generated": capture at user-provided decode step target_generated_step
            target_generated_step: Decode step index (1-based) used only with "specific_generated"
        
        Returns:
            Results dictionary with metadata
        """
        # Ensure model is loaded
        if self.model is None:
            self.load_model()

        # Normalize requested generation length
        try:
            requested_new_tokens = int(max_new_tokens)
        except (TypeError, ValueError):
            requested_new_tokens = 1
        requested_new_tokens = max(1, requested_new_tokens)
        
        allowed_modes = {"auto", "prefill", "random_generated", "specific_generated", "hybrid"}
        if capture_mode not in allowed_modes:
            raise ValueError(f"Invalid capture_mode='{capture_mode}'. Expected one of {sorted(allowed_modes)}")

        if capture_mode == "auto":
            effective_capture_mode = "hybrid" if requested_new_tokens > 1 else "prefill"
        else:
            effective_capture_mode = capture_mode
        
        # Tokenize input
        if self.verbose:
            print(f"{'='*70}")
            print(f"Running Inference")
            print(f"{'='*70}\n")
            print(f"Input text: '{text}'")
            print(f"Capture mode: {effective_capture_mode}")
        
        # Determine device for inputs
        device = next(self.model.parameters()).device
        tok_kwargs = {"return_tensors": "pt"}
        if max_seq_len is not None and max_seq_len > 0:
            tok_kwargs["truncation"] = True
            tok_kwargs["max_length"] = int(max_seq_len)
        inputs = self.tokenizer(text, **tok_kwargs).to(device)
        seq_len = inputs.input_ids.shape[1]
        token_ids = inputs.input_ids[0].tolist()
        
        if self.verbose:
            print(f"Tokens ({seq_len}): {token_ids}")
            if max_seq_len is not None and max_seq_len > 0:
                print(f"Max seq len setting: {max_seq_len}")
            decoded_tokens = [self.tokenizer.decode([tid]) for tid in token_ids]
            print(f"Decoded: {decoded_tokens}\n")

        # Clear cache before forward/decode pass.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        import time
        start_time = time.time()
        saved_count = 0
        predicted_token = ""
        generated_token_ids = []
        generated_text = ""
        generation_warning = None
        selected_generated_step = None
        captured_decode_step = None
        captured_seq_len = seq_len

        # Run inference/capture according to selected mode.
        if effective_capture_mode == "prefill":
            # For prefill, we usually capture the whole sequence (target_token_idx=None)
            self.register_hooks(num_layers, target_step_idx=0, target_token_idx=None)
            self.hook_manager.current_step = 0

            if self.verbose:
                print("Executing prefill forward pass...")
                if device.type == 'cpu':
                    print("⚠ Running on CPU - this may take several minutes...")

            try:
                with torch.no_grad():
                    outputs = self.model(**inputs)
            except torch.cuda.OutOfMemoryError as e:
                if self.verbose:
                    print(f"\n✗ CUDA OUT OF MEMORY during forward pass")
                    print(f"  Attempting emergency cache clearance...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise e

            forward_time = time.time() - start_time

            # First continuation token from prefill logits.
            next_token_id = torch.argmax(outputs.logits[0, -1, :]).item()
            predicted_token = self.tokenizer.decode([next_token_id])

            if self.verbose:
                print(f"✓ Forward pass complete ({forward_time:.2f} seconds)")
                print(f"  Predicted next token: '{predicted_token}'")

            activations = self.hook_manager.get_activations()

            if self.verbose:
                num_layers_captured = len(activations)
                num_activations = sum(len(layer_acts) for layer_acts in activations.values())
                print(f"Captured activations from {num_layers_captured} layers")
                print(f"Total activation tensors: {num_activations}\n")

                if activations:
                    first_layer = list(activations.keys())[0]
                    print(f"Sample (Layer {first_layer}):")
                    for name, tensor in activations[first_layer].items():
                        print(f"  {name}: {tuple(tensor.shape)}")
                    print()

            import os
            prefill_dir = os.path.join(output_dir, "prefill")
            saved_count = self.serializer.save_batch(activations, prefill_dir)
            self.hook_manager.remove_all_hooks()
            self.hook_manager.clear_activations()

            generated_token_ids = [next_token_id]
            generated_text = predicted_token

            if requested_new_tokens > 1:
                if self.verbose:
                    print(f"  Generating continuation (max_new_tokens={requested_new_tokens}, greedy decode)...")

                try:
                    generate_kwargs = {
                        "max_new_tokens": requested_new_tokens,
                        "do_sample": False,
                    }

                    pad_token_id = self.tokenizer.pad_token_id
                    if pad_token_id is None:
                        pad_token_id = self.tokenizer.eos_token_id
                    if pad_token_id is not None:
                        generate_kwargs["pad_token_id"] = pad_token_id
                    if self.tokenizer.eos_token_id is not None:
                        generate_kwargs["eos_token_id"] = self.tokenizer.eos_token_id

                    with torch.no_grad():
                        generated = self.model.generate(
                            input_ids=inputs.input_ids,
                            attention_mask=inputs.get("attention_mask"),
                            **generate_kwargs,
                        )

                    continuation_ids = generated[0, seq_len:].tolist()
                    if continuation_ids:
                        generated_token_ids = continuation_ids

                    generated_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
                    if generated_text == "":
                        generated_text = self.tokenizer.decode(generated_token_ids)

                    if self.verbose:
                        print(f"  ✓ Generated {len(generated_token_ids)} token(s)")

                except Exception as gen_exc:
                    generation_warning = str(gen_exc)
                    if self.verbose:
                        print(f"  ⚠ Multi-token generation failed; using single-token prediction ({generation_warning})")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            del outputs

        else:
            # ---------------------------------------------------------------
            # HYBRID APPROACH (Option 3): Generate all tokens first via
            # model.generate(), then replay ONE targeted forward pass with
            # hooks to capture activations at the chosen decode step.
            # This does exactly 2 forward-equivalent passes instead of N.
            # ---------------------------------------------------------------

            # 1. Pick the target decode step
            if effective_capture_mode == "specific_generated":
                if target_generated_step is None:
                    raise ValueError("target_generated_step is required when capture_mode='specific_generated'")
                selected_generated_step = int(target_generated_step)
                if selected_generated_step < 1 or selected_generated_step > requested_new_tokens:
                    raise ValueError(
                        f"target_generated_step must be in [1, {requested_new_tokens}], got {selected_generated_step}"
                    )
            else:
                selected_generated_step = random.randint(1, requested_new_tokens)

            if self.verbose:
                print(
                    f"Generating {requested_new_tokens} token(s) via model.generate(), "
                    f"then capturing activations at decode step {selected_generated_step}."
                )
                if device.type == 'cpu':
                    print("⚠ Running on CPU - generation may take a few minutes...")

            # 2. Fast generation — model.generate() uses its own internal
            #    KV-cache and is heavily optimized by HuggingFace.
            #    NO hooks are registered here so there is zero overhead.
            generate_kwargs = {
                "max_new_tokens": requested_new_tokens,
                "do_sample": False,
            }
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id
            if pad_token_id is not None:
                generate_kwargs["pad_token_id"] = pad_token_id
            if self.tokenizer.eos_token_id is not None:
                generate_kwargs["eos_token_id"] = self.tokenizer.eos_token_id

            try:
                with torch.no_grad():
                    generated = self.model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.get("attention_mask"),
                        **generate_kwargs,
                    )
            except torch.cuda.OutOfMemoryError as e:
                if self.verbose:
                    print(f"\n✗ CUDA OUT OF MEMORY during generation")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise e

            # Extract the generated continuation tokens
            all_generated_ids = generated[0, seq_len:].tolist()
            generated_token_ids = all_generated_ids
            if generated_token_ids:
                predicted_token = self.tokenizer.decode([generated_token_ids[0]])
                generated_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
                if generated_text == "":
                    generated_text = self.tokenizer.decode(generated_token_ids)

            gen_time = time.time() - start_time
            if self.verbose:
                print(f"  ✓ Generated {len(generated_token_ids)} token(s) in {gen_time:.2f}s")
                print(f"  Generated text: '{generated_text}'")

            # ----- HYBRID: Prefill activation capture (all prompt tokens) -----
            if effective_capture_mode == "hybrid":
                import os
                prefill_dir = os.path.join(output_dir, "prefill")
                os.makedirs(prefill_dir, exist_ok=True)

                if self.verbose:
                    print(f"\n  [HYBRID Phase 1] Capturing prefill activations (all {seq_len} prompt tokens)...")

                # Register hooks capturing all tokens (target_token_idx=None)
                self.register_hooks(num_layers, target_step_idx=0, target_token_idx=None)
                self.hook_manager.current_step = 0

                try:
                    with torch.no_grad():
                        prefill_outputs = self.model(**inputs)
                except torch.cuda.OutOfMemoryError as e:
                    if self.verbose:
                        print(f"\n✗ CUDA OUT OF MEMORY during prefill capture")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise e

                prefill_activations = self.hook_manager.get_activations()
                prefill_saved = self.serializer.save_batch(prefill_activations, prefill_dir)
                self.hook_manager.remove_all_hooks()
                self.hook_manager.clear_activations()

                if self.verbose:
                    print(f"  ✓ Prefill activations saved: {prefill_saved} files → {prefill_dir}/")
                    print(f"  [HYBRID Phase 2] Capturing decode step activations...")

                del prefill_outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Clamp selected step if generation produced fewer tokens than requested
            actual_gen_count = len(generated_token_ids)
            if actual_gen_count == 0:
                # Edge case: EOS was predicted immediately; fall back to prefill capture
                if self.verbose:
                    print("  ⚠ No tokens generated; falling back to prefill-style capture")
                self.register_hooks(num_layers, target_step_idx=0, target_token_idx=None)
                self.hook_manager.current_step = 0
                with torch.no_grad():
                    outputs = self.model(**inputs)
                activations = self.hook_manager.get_activations()
                import os
                prefill_dir = os.path.join(output_dir, "prefill")
                saved_count = self.serializer.save_batch(activations, prefill_dir)
                self.hook_manager.remove_all_hooks()
                self.hook_manager.clear_activations()
                captured_seq_len = seq_len
                del outputs
            else:
                if selected_generated_step > actual_gen_count:
                    selected_generated_step = actual_gen_count

                # 3. Build the context as it would have been at decode step k.
                #    At step k the model sees: [prompt_tokens] + [gen_token_1 ... gen_token_{k-1}]
                #    and predicts gen_token_k.
                context_ids = list(inputs.input_ids[0].tolist()) + generated_token_ids[:selected_generated_step - 1]
                capture_input_ids = torch.tensor([context_ids], device=device, dtype=inputs.input_ids.dtype)

                # Truncate to max_seq_len if needed
                max_context_len = int(max_seq_len) if max_seq_len is not None and max_seq_len > 0 else None
                if max_context_len is not None and capture_input_ids.shape[1] > max_context_len:
                    capture_input_ids = capture_input_ids[:, -max_context_len:]

                capture_attention_mask = torch.ones_like(capture_input_ids)

                # 4. Register hooks and run exactly ONE forward pass for capture.
                #    target_step_idx=0 and current_step=0 so hooks fire immediately.
                self.register_hooks(num_layers, target_step_idx=0, target_token_idx=-1)
                self.hook_manager.current_step = 0

                if self.verbose:
                    cap_start = time.time()
                    print(f"\n  Running 1 targeted forward pass for activation capture...")
                    print(f"  Context length: {capture_input_ids.shape[1]} tokens (prompt + {selected_generated_step - 1} generated)")

                try:
                    with torch.no_grad():
                        outputs = self.model(
                            input_ids=capture_input_ids,
                            attention_mask=capture_attention_mask,
                        )
                except torch.cuda.OutOfMemoryError as e:
                    if self.verbose:
                        print(f"\n✗ CUDA OUT OF MEMORY during capture forward pass")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise e

                if self.verbose:
                    print(f"  ✓ Capture forward pass complete ({time.time() - cap_start:.2f}s)")

                # 5. Save captured activations
                activations = self.hook_manager.get_activations()
                if activations:
                    first_layer = list(activations.keys())[0]
                    sample_tensor = next(iter(activations[first_layer].values()))
                    if sample_tensor.dim() >= 2:
                        captured_seq_len = int(sample_tensor.shape[-2])

                saved_count = self.serializer.save_batch(activations, output_dir)
                self.hook_manager.remove_all_hooks()
                self.hook_manager.clear_activations()
                captured_decode_step = selected_generated_step

                del outputs, capture_input_ids, capture_attention_mask

            if self.verbose:
                print(f"  Captured sequence length for proof: {captured_seq_len}")

        # Safety cleanup in case any hook remained active.
        if self.hook_manager.hooks:
            self.hook_manager.remove_all_hooks()
        self.hook_manager.clear_activations()

        # Explicit cleanup of large tensors
        del inputs

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        inference_time = time.time() - start_time

        if self.verbose:
            print(f"  Generated token count: {len(generated_token_ids)}")
            print(f"  Generated text: '{generated_text}'")
            print(f"✓ Inference complete ({inference_time:.2f} seconds)\n")
        
        # NOTE: We skip explicit 'del self.model' here because this manager is 
        # typically used in a standalone subprocess. Explicit deletion can 
        # sometimes trigger segfaults in bitsandbytes/CUDA destructors 
        # during process exit. The OS will reclaim all VRAM/RAM on exit.
        
        result = {
            'input_text': text,
            'predicted_token': predicted_token,
            'generated_text': generated_text,
            'generated_token_ids': generated_token_ids,
            'num_generated_tokens': len(generated_token_ids),
            'max_new_tokens_requested': requested_new_tokens,
            'capture_mode': effective_capture_mode,
            'captured_seq_len': captured_seq_len,
            'generation_warning': generation_warning,
            'seq_len': captured_seq_len,
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
            print(f"Generated token count: {len(generated_token_ids)}")
            print(f"Generated text: '{generated_text}'")
            print(f"Captured seq_len: {captured_seq_len}")
            print(f"Inference time: {inference_time:.2f}s")
            print(f"Activation files saved: {saved_count}")
            print(f"Output directory: {output_dir}/")
            print(f"{'='*70}\n")
        
        return result