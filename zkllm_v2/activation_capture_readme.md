# Activation Capture System for zkLLM

## Overview

This activation capture system enables the extraction of intermediate activations from LLaMA-2 transformer layers during inference. These activations serve as **real input data** for zkLLM's zero-knowledge proof generation pipeline, replacing the random dummy data used in testing.

## Table of Contents

- [LLaMA-2 Architecture Background](#llama-2-architecture-background)
- [Activation Capture Architecture](#activation-capture-architecture)
- [Detailed Flow with Analogy](#detailed-flow-with-analogy)
- [File Format Specification](#file-format-specification)
- [Usage Guide](#usage-guide)
- [Integration with Proof Generation](#integration-with-proof-generation)
- [Technical Details](#technical-details)

---

## LLaMA-2 Architecture Background

### Transformer Block Structure

Each LLaMA-2 layer is a **transformer decoder block** with the following components:

```
┌─────────────────────────────────────────────────┐
│           LLaMA Transformer Layer               │
├─────────────────────────────────────────────────┤
│                                                 │
│  Input (x) ──┐                                 │
│              │                                  │
│              ├─► Input RMSNorm                  │
│              │         │                        │
│              │         ▼                        │
│              │   Self-Attention                 │
│              │    (Q, K, V, O)                  │
│              │         │                        │
│              └────────(+)  [Skip Connection 1]  │
│                       │                         │
│                       ├─► Post-Attn RMSNorm     │
│                       │         │               │
│                       │         ▼               │
│                       │    Feed-Forward         │
│                       │   (Gate, Up, Down)      │
│                       │         │               │
│                       └────────(+) [Skip Conn 2]│
│                                │                │
│                          Output (x')            │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Component Details

1. **RMSNorm (Root Mean Square Normalization)**
   - Normalizes activations to stabilize training/inference
   - Formula: `x_norm = x / sqrt(mean(x²) + ε)`
   - Applied twice: before self-attention and before FFN

2. **Self-Attention**
   - Projects input to Query (Q), Key (K), Value (V)
   - Computes attention: `softmax(QK^T / √d) @ V`
   - Projects output with O matrix
   - Uses RoPE (Rotary Position Embeddings)

3. **Feed-Forward Network (FFN)**
   - Three linear transformations: Gate, Up, Down
   - Activation: SwiGLU = `Gate(x) * σ(Up(x))`
   - Projects back to embedding dimension

4. **Skip Connections (Residual)**
   - First: `x + self_attn(RMSNorm(x))`
   - Second: `x + FFN(RMSNorm(x))`
   - Enables gradient flow and stable training

---

## Activation Capture Architecture

### Design Philosophy

The capture system uses **PyTorch forward hooks** to intercept intermediate activations at specific computation points without modifying the model's execution. Think of it as placing "sensors" at critical junctions in the data pipeline.

### Hook Placement Strategy

We capture **6 critical activation points** per layer:

```
┌──────────────────────────────────────────────────────────────┐
│                    Activation Capture Points                  │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  [1] block_input ◄──── Input to entire layer                │
│        │                                                     │
│        ▼                                                     │
│  Input RMSNorm                                               │
│        │                                                     │
│  [2] input_layernorm_output ◄──── After input norm          │
│        │                                                     │
│        ▼                                                     │
│  Self-Attention (Q, K, V, O)                                 │
│        │                                                     │
│        ▼                                                     │
│  (Skip Connection 1: block_input + attn_output)              │
│        │                                                     │
│  [3] post_attn_residual ◄──── After first skip              │
│        │                                                     │
│        ▼                                                     │
│  Post-Attention RMSNorm                                      │
│        │                                                     │
│  [4] ffn_input ◄──── After post-attn norm                   │
│        │                                                     │
│        ▼                                                     │
│  [5] mlp_output ◄──── FFN output                            │
│        │                                                     │
│        ▼                                                     │
│  (Skip Connection 2: post_attn_residual + mlp_output)        │
│        │                                                     │
│  [6] block_output ◄──── Final layer output                  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Why These 6 Points?

Each capture point corresponds to an **input required by zkLLM proof components**:

| Capture Point | Proof Component | Purpose |
|--------------|-----------------|---------|
| `block_input` | Input RMSNorm, Skip Connection | Input to normalization + skip proof |
| `input_layernorm_output` | Self-Attention | Input to Q, K, V projections |
| `post_attn_residual` | Post-Attn RMSNorm | Input to second normalization |
| `ffn_input` | FFN | Input to feed-forward network |
| `mlp_output` | (intermediate) | For computing final skip |
| `block_output` | Skip Connection | Output after all computations |

---

## Detailed Flow with Analogy

### Analogy: Factory Assembly Line

Imagine the LLaMA model as a **factory assembly line** where each layer is a **production station**. The activation capture system acts as **quality control checkpoints** that photograph the product at specific stages.

#### Stage 1: Raw Material Arrives (Block Input)

```python
# Hook 1: Pre-hook on layer module
def capture_block_input(module, input):
    hidden_states = input[0]  # Extract tensor from tuple
    self.activations[layer_idx]['block_input'] = hidden_states.detach().clone()
```

**Analogy**: A **raw material inspector** photographs the parts entering the production station.

**What happens**:
- Input tensor shape: `[batch_size, seq_len, hidden_dim]` = `[1, 128, 4096]`
- PyTorch pre-hook intercepts the input **before** any processing
- We extract `hidden_states` (the actual data, not metadata)
- `.detach()` removes gradient tracking (we only need values, not gradients)
- `.clone()` creates a copy (prevents reference issues)

**Saved as**: `layer-{i}-input-rmsnorm-activation.bin`

---

#### Stage 2: First Quality Check (Input RMSNorm Output)

```python
# Hook 2: Forward hook on input_layernorm
def capture_input_norm_output(module, input, output):
    self.activations[layer_idx]['input_layernorm_output'] = output.detach().clone()
```

**Analogy**: After the **cleaning station**, a supervisor verifies the parts are properly cleaned and normalized.

**What happens**:
- RMSNorm normalizes each token vector to unit scale
- Formula: `output = input / sqrt(mean(input²) + eps) * weight`
- Output shape: Same as input `[1, 128, 4096]`
- Forward hook captures **after** RMSNorm computes its output
- This becomes the input to Q, K, V projections in self-attention

**Saved as**: `layer-{i}-self-attn-activation.bin`

**zkLLM Usage**: The proof verifies:
1. RMSNorm computation was correct
2. Q, K, V projections use this normalized input

---

#### Stage 3: Assembly Complete (Post-Attention Residual)

```python
# Hook 3: Pre-hook on post_attention_layernorm
def capture_post_attn_residual(module, input):
    hidden_states = input[0]  # This is residual + attn_output
    self.activations[layer_idx]['post_attn_residual'] = hidden_states.detach().clone()
```

**Analogy**: After the **assembly robot** (self-attention) adds components, a **merger checkpoint** verifies the combined result.

**What happens**:
- Self-attention computes attention and produces `attn_output`
- Skip connection adds: `residual = block_input + attn_output`
- This is the input to `post_attention_layernorm`
- Pre-hook on post-attn norm captures this **merged state**
- Shape: Still `[1, 128, 4096]`

**Saved as**: `layer-{i}-post_attention-rmsnorm-activation.bin`

**Mathematical Detail**:
```
block_input:         [1, 128, 4096]
    ↓
input_layernorm(x):  [1, 128, 4096]
    ↓
self_attn(x):        [1, 128, 4096]
    ↓
block_input + attn:  [1, 128, 4096] ← CAPTURED HERE
```

---

#### Stage 4: Second Cleaning (FFN Input)

```python
# Hook 4: Forward hook on post_attention_layernorm
def capture_ffn_input(module, input, output):
    self.activations[layer_idx]['ffn_input'] = output.detach().clone()
```

**Analogy**: Before the **painting station** (FFN), parts go through a **second cleaning booth** to ensure smooth coating.

**What happens**:
- Post-attention RMSNorm normalizes the residual
- Formula: Same RMSNorm computation as before
- Output becomes input to FFN (gate_proj, up_proj, down_proj)
- Shape: `[1, 128, 4096]`

**Saved as**: `layer-{i}-ffn-activation.bin`

**zkLLM Usage**: Proves FFN computation starts from correctly normalized input

---

#### Stage 5: Painting Done (MLP Output)

```python
# Hook 5: Forward hook on mlp module
def capture_mlp_output(module, input, output):
    self.activations[layer_idx]['mlp_output'] = output.detach().clone()
```

**Analogy**: After the **painting station** finishes, a camera captures the newly painted product.

**What happens**:
- FFN applies three transformations:
  - `gate = gate_proj(x)`  # [1, 128, 4096] → [1, 128, 11008]
  - `up = up_proj(x)`      # [1, 128, 4096] → [1, 128, 11008]
  - `activated = gate * silu(up)`  # SwiGLU activation
  - `output = down_proj(activated)` # [1, 128, 11008] → [1, 128, 4096]
- Output shape: `[1, 128, 4096]`
- This will be added to `post_attn_residual` in the second skip

**Note**: We capture this for computing `block_output`, but don't save it separately (it's an intermediate value).

---

#### Stage 6: Final Product (Block Output)

```python
# Hook 6: Forward hook on entire layer module
def capture_block_output(module, input, output):
    if isinstance(output, tuple):
        hidden_states = output[0]  # Extract from tuple
    else:
        hidden_states = output
    self.activations[layer_idx]['block_output'] = hidden_states.detach().clone()
```

**Analogy**: The **final inspection station** photographs the completed product leaving the production line.

**What happens**:
- Final skip connection: `output = post_attn_residual + mlp_output`
- This output becomes the input to the next layer
- Some models return tuples (hidden_states, attention_weights, etc.)
- We extract just the hidden_states
- Shape: `[1, 128, 4096]`

**Saved as**: `layer-{i}-block-output.bin`

**zkLLM Usage**: Proves the second skip connection was computed correctly:
```
block_output = post_attn_residual + mlp_output
```

---

## File Format Specification

### Binary Format

All activation files use the **int32 fixed-point** format required by zkLLM:

```
┌─────────────────────────────────────┐
│     Activation Binary File          │
├─────────────────────────────────────┤
│ No header                           │
│ Raw int32 array                     │
│ Row-major order (C-style)           │
│ Shape: (seq_len, hidden_dim)        │
│ Scale factor: 2^16 = 65536          │
└─────────────────────────────────────┘
```

### Conversion Process

```python
# 1. Start with PyTorch tensor (float32)
tensor = torch.tensor([...])  # Shape: [batch, seq_len, hidden_dim]

# 2. Remove batch dimension (always 1 during inference)
tensor = tensor.squeeze(0)  # Shape: [seq_len, hidden_dim]

# 3. Move to CPU and convert to numpy
tensor_float = tensor.cpu().float().numpy()

# 4. Scale by 2^16 and round to nearest integer
scale_factor = 2 ** 16  # = 65536
tensor_int = np.round(tensor_float * scale_factor).astype(np.int32)

# 5. Save as raw binary (no header, just data)
tensor_int.tofile(filepath)
```

### File Size Calculation

For `seq_len=128`, `hidden_dim=4096`:

```
Elements per file = seq_len × hidden_dim = 128 × 4096 = 524,288
Bytes per file = 524,288 × 4 bytes = 2,097,152 bytes = 2 MB
Files per layer = 6
Total per layer = 6 × 2 MB = 12 MB
Total for 32 layers = 32 × 12 MB = 384 MB
```

### Example: Reading a File

```python
import numpy as np

# Read the binary file
data = np.fromfile('activations/layer-0-input-rmsnorm-activation.bin', 
                   dtype=np.int32)

# Reshape to (seq_len, hidden_dim)
seq_len, hidden_dim = 128, 4096
activations = data.reshape(seq_len, hidden_dim)

# Convert back to float (descale)
activations_float = activations.astype(np.float32) / (2 ** 16)

print(f"Shape: {activations.shape}")
print(f"Range: [{activations.min()}, {activations.max()}]")
print(f"Mean: {activations.mean():.2f}")
```

---

## Usage Guide

### Basic Usage

```bash
# 1. Activate the conda environment
conda activate zkllm-env

# 2. Capture activations for all 32 layers
python capture_activations.py \
    --text "Your input text here" \
    --num_layers 32 \
    --output_dir activations/

# 3. The script will tell you the exact seq_len to use
# Example output:
# ⚠️  IMPORTANT: Use this command for proof generation:
# python generate_proofs.py --seq_len 128 --start_layer 0 --end_layer 31
```

### Advanced Options

```bash
# Capture only specific layers (e.g., first 5 layers)
python capture_activations.py \
    --text "Test input" \
    --num_layers 5 \
    --output_dir activations/

# Use CPU-only mode (no GPU)
python capture_activations.py \
    --text "Test input" \
    --cpu

# Quiet mode (less verbose output)
python capture_activations.py \
    --text "Test input" \
    --quiet
```

### Generating Text with Specific Token Count

The `seq_len` parameter in proof generation **must match** the token count of your input text. Here's how to generate text with a specific token count:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# For seq_len = 128, you need 127 tokens (+ 1 BOS token)
text = " ".join(["Hello"] * 127)
tokens = tokenizer.encode(text)
print(f"Token count: {len(tokens)}")  # Should be 128

# Use this text for capture
python capture_activations.py --text "$text" --num_layers 32
```

---

## Integration with Proof Generation

### File Mapping

The capture system creates files that **exactly match** what `generate_proofs.py` expects:

| Capture Output | Proof Script | Component |
|----------------|--------------|-----------|
| `layer-{i}-input-rmsnorm-activation.bin` | `llama-rmsnorm.py` | Input RMSNorm |
| `layer-{i}-self-attn-activation.bin` | `llama-self-attn.py` | Self-Attention |
| `layer-{i}-post_attention-rmsnorm-activation.bin` | `llama-rmsnorm.py` | Post-Attn RMSNorm |
| `layer-{i}-ffn-activation.bin` | `llama-ffn.py` | Feed-Forward Network |
| `layer-{i}-block-input.bin` | `llama-skip-connection.py` | Skip Connection 1 |
| `layer-{i}-block-output.bin` | `llama-skip-connection.py` | Skip Connection 2 |

### Proof Generation Workflow

```bash
# Step 1: Capture activations
python capture_activations.py --text "..." --num_layers 32
# Output: Creates 192 files (6 per layer × 32 layers)

# Step 2: Generate proofs for all layers
python generate_proofs.py --seq_len <TOKEN_COUNT> --start_layer 0 --end_layer 31

# Or test a single layer first
python generate_proofs.py --seq_len <TOKEN_COUNT> --single_layer 0
```

### What Happens During Proof Generation

1. **RMSNorm Proof** (`llama-rmsnorm.py`):
   - Reads: `layer-{i}-input-rmsnorm-activation.bin`
   - Computes: RMSNorm using model weights
   - Generates: Proof that normalization was correct

2. **Self-Attention Proof** (`llama-self-attn.py`):
   - Reads: `layer-{i}-self-attn-activation.bin`
   - Computes: Q, K, V projections, attention, O projection
   - Generates: Proof for each linear layer and attention computation

3. **FFN Proof** (`llama-ffn.py`):
   - Reads: `layer-{i}-ffn-activation.bin`
   - Computes: Gate, Up, Down projections with SwiGLU
   - Generates: Proof for each FFN component

4. **Skip Connection Proof** (`llama-skip-connection.py`):
   - Reads: `layer-{i}-block-input.bin` and `layer-{i}-block-output.bin`
   - Verifies: `block_output = block_input + mlp_output`
   - Generates: Proof that residual connection is correct

---

## Technical Details

### Memory Optimization: 4-bit Quantization

To fit LLaMA-2-7B in 6GB GPU memory, we use **4-bit quantization** during capture:

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_card,
    quantization_config=quantization_config,
    device_map="auto"
)
```

**Important**: Activations are captured in **full precision** (float32) before quantization, so proof accuracy is not affected.

### Hook Lifecycle

```python
# 1. Register hooks before inference
hook_manager.register_layer_hooks(layer_idx=0, layer_module=model.model.layers[0])

# 2. Run inference (hooks automatically capture)
with torch.no_grad():
    outputs = model(input_ids)

# 3. Retrieve captured activations
activations = hook_manager.get_activations()

# 4. Save to disk
serializer.save_layer_activations(activations[0], layer_idx=0)

# 5. Clean up hooks
hook_manager.remove_all_hooks()
```

### Why `.detach().clone()`?

```python
# BAD: Just reference
self.activations[i] = output  # ❌ Reference to computation graph

# GOOD: Detach and clone
self.activations[i] = output.detach().clone()  # ✅ Independent copy
```

**Reasons**:
- `.detach()`: Removes from autograd graph (no gradient tracking)
- `.clone()`: Creates new memory allocation (prevents sharing)
- Without these, activations might be overwritten or cause memory leaks

### Sequence Length Constraints

The zkLLM CUDA binaries have constraints on sequence length:

```
Minimum seq_len: ~64 (power of 2 preferred)
Recommended: 128 (good balance of speed and accuracy)
Maximum tested: 2048 (from original paper)
```

**Why?** The CUDA implementation uses:
- Power-of-2 optimized FFT for polynomial operations
- Fixed-size lookup tables for softmax
- Memory layouts optimized for specific sizes

---

## Troubleshooting

### Issue: "cannot reshape array"

```
ValueError: cannot reshape array of size 266240 into shape (128,4096)
```

**Cause**: `seq_len` mismatch between capture and proof generation.

**Solution**: Use the exact `seq_len` printed by `capture_activations.py`:
```bash
# Capture prints: "Sequence length: 65 tokens"
# Use this in proof generation:
python generate_proofs.py --seq_len 65 ...
```

### Issue: "D or N is not power of 2"

```
terminate called after throwing an instance of 'std::runtime_error'
what():  D or N is not power of 2, or D is not divisible by N
```

**Cause**: Sequence length too small (< 64).

**Solution**: Use at least `seq_len=64`, preferably `seq_len=128`.

### Issue: Out of GPU memory

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solution**: The capture system automatically uses 4-bit quantization. If still failing:
```bash
# Use CPU-only mode
python capture_activations.py --cpu --text "..."
```

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                    Full Pipeline Overview                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. USER INPUT                                                   │
│     ├─► "Hello world this is a test"                            │
│     └─► Tokenized: [1, 15043, 3186, 445, 338, 263, 1243]        │
│                            │                                     │
│  2. ACTIVATION CAPTURE                                           │
│     ├─► Load LLaMA-2-7B (4-bit quantized)                       │
│     ├─► Register 6 hooks per layer                              │
│     ├─► Run inference with hooks                                │
│     └─► Save 192 files (6 × 32 layers)                          │
│                            │                                     │
│  3. PROOF GENERATION                                             │
│     ├─► Read activation files                                   │
│     ├─► For each layer:                                         │
│     │   ├─► RMSNorm proof                                       │
│     │   ├─► Self-Attention proof                                │
│     │   ├─► Post-Attn RMSNorm proof                             │
│     │   ├─► FFN proof                                           │
│     │   └─► Skip Connection proof                               │
│     └─► Generate 160 proof files (5 × 32 layers)                │
│                            │                                     │
│  4. VERIFICATION                                                 │
│     └─► Verifier checks proofs against public parameters        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Summary

The activation capture system acts as a **bridge** between the LLaMA-2 model and zkLLM's proof generation:

1. **Intercepts** intermediate computations using PyTorch hooks
2. **Extracts** activations at 6 critical points per layer
3. **Converts** float tensors to int32 fixed-point format
4. **Saves** in binary files with zkLLM-compatible naming
5. **Enables** proof generation with real inference data

This replaces random dummy data with **authentic model activations**, making the zero-knowledge proofs verify actual model computations rather than synthetic data.

---

## References

- [zkLLM Paper (ACM CCS 2024)](https://arxiv.org/abs/2404.16109)
- [LLaMA-2 Paper](https://arxiv.org/abs/2307.09288)
- [PyTorch Hooks Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook)
- [Transformers Library](https://huggingface.co/docs/transformers)
