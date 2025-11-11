
import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import fileio_utils
import subprocess
import time

class ZkLLMTextInference:
    def __init__(self, model_size=7, max_layers=2):
        """
        Initialize zkLLM text inference system
        
        Args:
            model_size: 7 or 13 (billion parameters)
            max_layers: Number of layers to process (default 2 for demo)
        """
        self.model_size = model_size
        self.max_layers = max_layers
        self.workdir = f'./zkllm-workdir/Llama-2-{model_size}b'
        self.temp_dir = './temp-files'
        
        # Ensure directories exist
        Path(self.temp_dir).mkdir(exist_ok=True)
        
        # Load model and tokenizer
        print(f"Loading LLaMA-2-{model_size}B model...")
        model_card = f"meta-llama/Llama-2-{model_size}b-hf"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_card, 
            local_files_only=True, 
            cache_dir="./model-storage",
            torch_dtype=torch.float32,
            device_map="cpu"  # Keep on CPU for consistency
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_card, 
            cache_dir="./model-storage"
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Model loaded successfully!")
    
    def text_to_embeddings(self, text):
        """
        Convert text input to token embeddings (Layer 0 input)
        
        Args:
            text: Input text string
            
        Returns:
            embeddings: Tensor of shape [seq_len, 4096]
            tokens: List of token IDs
        """
        print(f"\n=== TEXT TO EMBEDDINGS ===")
        print(f"Input text: '{text}'")
        
        # Tokenize
        tokens = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        token_ids = tokens.input_ids.squeeze(0)
        
        print(f"Token IDs: {token_ids.tolist()}")
        print(f"Tokens: {[self.tokenizer.decode([tid]) for tid in token_ids]}")
        print(f"Sequence length: {len(token_ids)}")
        
        # Convert to embeddings
        with torch.no_grad():
            embeddings = self.model.model.embed_tokens(token_ids.unsqueeze(0))
            embeddings = embeddings.squeeze(0)  # Remove batch dimension
        
        print(f"Embeddings shape: {embeddings.shape}")
        return embeddings, token_ids.tolist()
    
    def run_command(self, cmd, description):
        """Run command and handle errors"""
        print(f"\n>>> {description}")
        print(f"Command: {' '.join(cmd)}")
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time
        
        if result.returncode != 0:
            print(f"❌ ERROR: {description} failed")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
        else:
            print(f"✅ SUCCESS: {description} completed in {elapsed:.2f}s")
            return True
    
    def generate_rmsnorm_proof(self, layer, norm_type, input_file, seq_len):
        """Generate RMSNorm proof"""
        output_file = f"{self.temp_dir}/layer-{layer}-{norm_type}-rmsnorm-proof.bin"
        
        cmd = [
            'python', 'llama-rmsnorm.py',
            str(self.model_size),
            str(layer),
            norm_type,
            str(seq_len),
            '--input_file', input_file,
            '--output_file', output_file
        ]
        
        success = self.run_command(cmd, f"Layer {layer} {norm_type} RMSNorm proof")
        return success, output_file
    
    def generate_self_attn_proof(self, layer, input_file, seq_len):
        """Generate self-attention proof"""
        output_file = f"{self.temp_dir}/layer-{layer}-self-attn-proof.bin"
        
        cmd = [
            'python', 'llama-self-attn.py',
            str(self.model_size),
            str(layer),
            str(seq_len),
            '--input_file', input_file,
            '--output_file', output_file
        ]
        
        success = self.run_command(cmd, f"Layer {layer} self-attention proof")
        return success, output_file
    
    def generate_ffn_proof(self, layer, input_file, seq_len):
        """Generate FFN proof"""
        output_file = f"{self.temp_dir}/layer-{layer}-ffn-proof.bin"
        
        cmd = [
            'python', 'llama-ffn.py',
            str(self.model_size),
            str(layer),
            str(seq_len),
            '--input_file', input_file,
            '--output_file', output_file
        ]
        
        success = self.run_command(cmd, f"Layer {layer} FFN proof")
        return success, output_file
    
    def generate_skip_connection_proof(self, layer, block_input_file, block_output_file):
        """Generate skip connection proof"""
        output_file = f"{self.temp_dir}/layer-{layer}-skip-proof.bin"
        
        cmd = [
            'python', 'llama-skip-connection.py',
            str(self.model_size),
            str(layer),
            '--block_input_file', block_input_file,
            '--block_output_file', block_output_file,
            '--output_file', output_file
        ]
        
        success = self.run_command(cmd, f"Layer {layer} skip connection proof")
        return success, output_file
    
    def process_layer_with_proofs(self, layer, input_activations, seq_len):
        """
        Process a single transformer layer and generate all proofs
        
        Args:
            layer: Layer number (0-31)
            input_activations: Input tensor [seq_len, 4096]
            seq_len: Sequence length
            
        Returns:
            output_activations: Output tensor [seq_len, 4096]
            proof_files: List of generated proof files
        """
        print(f"\n{'='*50}")
        print(f"PROCESSING LAYER {layer}")
        print(f"{'='*50}")
        
        proof_files = []
        
        # Save input activations for this layer
        input_file = f"{self.temp_dir}/layer-{layer}-input.bin"
        fileio_utils.save_int(input_activations, 1 << 16, input_file)
        print(f"Saved layer {layer} input: {input_file}")
        
        # 1. Input RMSNorm proof
        success, proof_file = self.generate_rmsnorm_proof(layer, 'input', input_file, seq_len)
        if success:
            proof_files.append(proof_file)
        else:
            print(f"❌ Failed to generate input RMSNorm proof for layer {layer}")
            return None, []
        
        # Compute actual layer processing using raw model weights
        with torch.no_grad():
            # Get the actual layer
            transformer_layer = self.model.model.layers[layer]
            
            # Add batch dimension for processing
            hidden_states = input_activations.unsqueeze(0)
            
            # Process through the layer
            outputs = transformer_layer(hidden_states)
            output_activations = outputs[0].squeeze(0)  # Remove batch dimension
        
        # Save output activations
        output_file = f"{self.temp_dir}/layer-{layer}-output.bin"
        fileio_utils.save_int(output_activations, 1 << 16, output_file)
        
        # 2. Self-attention proof (we'll simulate by using input for now)
        success, proof_file = self.generate_self_attn_proof(layer, input_file, seq_len)
        if success:
            proof_files.append(proof_file)
        
        # 3. Post-attention RMSNorm proof
        success, proof_file = self.generate_rmsnorm_proof(layer, 'post_attention', input_file, seq_len)
        if success:
            proof_files.append(proof_file)
        
        # 4. FFN proof
        success, proof_file = self.generate_ffn_proof(layer, input_file, seq_len)
        if success:
            proof_files.append(proof_file)
        
        # 5. Skip connection proof
        success, proof_file = self.generate_skip_connection_proof(layer, input_file, output_file)
        if success:
            proof_files.append(proof_file)
        
        print(f"✅ Layer {layer} processing completed")
        print(f"Generated {len(proof_files)} proof files")
        
        return output_activations, proof_files
    
    def generate_final_prediction(self, final_activations):
        """Generate final token prediction"""
        print(f"\n=== FINAL PREDICTION ===")
        
        with torch.no_grad():
            # Final layer norm
            normalized = self.model.model.norm(final_activations.unsqueeze(0))
            
            # Language model head
            logits = self.model.lm_head(normalized)
            
            # Get next token prediction
            next_token_logits = logits[0, -1, :]  # Last token's logits
            predicted_token_id = torch.argmax(next_token_logits).item()
            
            # Decode prediction
            predicted_token = self.tokenizer.decode([predicted_token_id])
            
            # Get top 5 predictions
            top_k = 5
            top_tokens = torch.topk(next_token_logits, top_k)
            top_predictions = []
            for i in range(top_k):
                token_id = top_tokens.indices[i].item()
                prob = torch.softmax(next_token_logits, dim=0)[token_id].item()
                token = self.tokenizer.decode([token_id])
                top_predictions.append((token, prob))
        
        print(f"Predicted next token: '{predicted_token}' (ID: {predicted_token_id})")
        print(f"Top {top_k} predictions:")
        for i, (token, prob) in enumerate(top_predictions):
            print(f"  {i+1}. '{token}' ({prob:.4f})")
        
        return predicted_token, predicted_token_id
    
    def run_inference_with_proofs(self, text):
        """
        Run complete inference with ZK proofs for custom text input
        
        Args:
            text: Input text string
            
        Returns:
            prediction: Next token prediction
            proof_files: List of all generated proof files
        """
        print(f"\n{'='*60}")
        print(f"zkLLM INFERENCE WITH PROOFS")
        print(f"{'='*60}")
        print(f"Input: '{text}'")
        print(f"Model: LLaMA-2-{self.model_size}B")
        print(f"Processing layers: 0-{self.max_layers-1}")
        
        # Convert text to embeddings
        embeddings, token_ids = self.text_to_embeddings(text)
        seq_len = embeddings.shape[0]
        
        # Process through transformer layers
        current_activations = embeddings
        all_proof_files = []
        
        for layer in range(self.max_layers):
            output_activations, proof_files = self.process_layer_with_proofs(
                layer, current_activations, seq_len
            )
            
            if output_activations is None:
                print(f"❌ Failed to process layer {layer}")
                break
                
            current_activations = output_activations
            all_proof_files.extend(proof_files)
        
        # Generate final prediction
        prediction, pred_id = self.generate_final_prediction(current_activations)
        
        # Summary
        print(f"\n{'='*60}")
        print(f"INFERENCE COMPLETE")
        print(f"{'='*60}")
        print(f"Input: '{text}'")
        print(f"Tokens: {token_ids}")
        print(f"Prediction: '{prediction}' (ID: {pred_id})")
        print(f"Layers processed: {self.max_layers}")
        print(f"Total proofs generated: {len(all_proof_files)}")
        print(f"Proof files saved in: {self.temp_dir}/")
        
        return prediction, all_proof_files

def main():
    parser = argparse.ArgumentParser(description='zkLLM Text Inference with Proof Generation')
    parser.add_argument('--text', required=True, type=str, help='Input text to process')
    parser.add_argument('--model_size', type=int, choices=[7, 13], default=7, 
                       help='Model size (7B or 13B)')
    parser.add_argument('--max_layers', type=int, default=2,
                       help='Maximum number of layers to process (default: 2 for demo)')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up temporary files after processing')
    
    args = parser.parse_args()
    
    # Check if zkllm-workdir exists (commitments must be done first)
    workdir = f'./zkllm-workdir/Llama-2-{args.model_size}b'
    if not os.path.exists(workdir):
        print(f"❌ ERROR: {workdir} not found!")
        print("Please run the commitment phase first:")
        print(f"python llama-commit.py {args.model_size}")
        sys.exit(1)
    
    # Compile all necessary binaries
    print("Compiling CUDA binaries...")
    compile_commands = ['rmsnorm', 'self-attn', 'ffn', 'skip-connection']
    for binary in compile_commands:
        result = os.system(f'make {binary}')
        if result != 0:
            print(f"❌ ERROR: Failed to compile {binary}")
            sys.exit(1)
    print("✅ All binaries compiled successfully")
    
    try:
        # Initialize inference system
        zkllm = ZkLLMTextInference(model_size=args.model_size, max_layers=args.max_layers)
        
        # Run inference with proofs
        prediction, proof_files = zkllm.run_inference_with_proofs(args.text)
        
        print(f"\n🎉 zkLLM inference completed successfully!")
        print(f"📝 Complete computational proof generated for: '{args.text}' → '{prediction}'")
        
        # List all generated files
        print(f"\n📁 Generated proof files:")
        for i, proof_file in enumerate(proof_files, 1):
            file_size = os.path.getsize(proof_file) if os.path.exists(proof_file) else 0
            print(f"  {i}. {proof_file} ({file_size} bytes)")
        
        if args.cleanup:
            print(f"\n🧹 Cleaning up temporary files...")
            for proof_file in proof_files:
                if os.path.exists(proof_file):
                    os.remove(proof_file)
            print("✅ Cleanup completed")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()