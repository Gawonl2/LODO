"""
LLaMA Model Wrapper for Local GPU Inference

This module provides functions to load and run LLaMA-style models on GPU.
Designed for use with NVIDIA GPUs (tested on RTX 3090).

GPU Memory Requirements:
- Llama-3.1-8B-Instruct: ~16GB VRAM (with bfloat16)
- For 24GB GPUs (RTX 3090), 8B models work well with bfloat16
- If you have less VRAM, consider using 4-bit quantization or smaller models

Model Access:
- Llama-3.1-8B-Instruct requires approval from Meta
- Visit: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- Alternative open models: microsoft/Phi-3-mini-4k-instruct, Qwen/Qwen2-7B-Instruct
"""

import torch
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional

# Try to import huggingface_hub for authentication
try:
    from huggingface_hub import login, whoami, snapshot_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("Warning: huggingface_hub not available. Install with: pip install huggingface_hub")

# Try to set token from file early (before any imports that might need it)
def _set_token_early():
    """Set token from file early so transformers can use it."""
    script_dir = Path(__file__).parent
    possible_files = [
        script_dir / "hg.txt",
        script_dir / "hg",
        Path.cwd() / "hg.txt",
        Path.cwd() / "hg",
    ]
    
    for file_path in possible_files:
        if file_path.exists():
            try:
                token = file_path.read_text().strip()
                if token and token.startswith("hf_"):
                    os.environ["HF_TOKEN"] = token
                    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
                    return token
            except Exception:
                pass
    return None

# Set token early if available
_set_token_early()

# ==================== Configuration ====================
# TODO: Change this to another model name if needed
# Examples:
#   "meta-llama/Llama-3.1-8B-Instruct" (requires Meta approval, default)
#   "meta-llama/Llama-3-8B-Instruct" (requires Meta approval)
#   "microsoft/Phi-3-mini-4k-instruct" (open, smaller ~3.8B)
#   "Qwen/Qwen2-7B-Instruct" (open, Apache 2.0 license)
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# Generation parameters (can be overridden in generate_text())
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.7  # Lower = more deterministic, Higher = more creative
DEFAULT_TOP_P = 0.9  # Nucleus sampling threshold
# ======================================================

# Global model and tokenizer (loaded once, reused for all generations)
_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForCausalLM] = None
_loaded_model_name: Optional[str] = None
_authenticated: bool = False
_hf_token: Optional[str] = None  # Store token for explicit use


def _get_token_from_file() -> Optional[str]:
    """Read Hugging Face token from hg.txt or hg file."""
    # Try multiple possible file locations
    script_dir = Path(__file__).parent
    possible_files = [
        script_dir / "hg.txt",
        script_dir / "hg",
        Path.cwd() / "hg.txt",
        Path.cwd() / "hg",
    ]
    
    for file_path in possible_files:
        if file_path.exists():
            try:
                token = file_path.read_text().strip()
                if token and token.startswith("hf_"):
                    print(f"Found token in {file_path.name}")
                    return token
            except Exception as e:
                print(f"Warning: Could not read token from {file_path}: {e}")
    
    return None


def ensure_authentication():
    """
    Ensure Hugging Face authentication is set up.
    Reads token from hg.txt or hg file and authenticates.
    """
    global _authenticated, _hf_token
    
    if _authenticated:
        return True
    
    if not HF_HUB_AVAILABLE:
        print("Warning: Cannot authenticate - huggingface_hub not available")
        return False
    
    # Check if already authenticated
    try:
        user_info = whoami()
        print(f"Already authenticated as: {user_info.get('name', 'Unknown')}")
        _authenticated = True
        return True
    except Exception:
        pass  # Not authenticated, continue
    
    # Try to get token from environment variable first
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    
    # If not in environment, try reading from file
    if not token:
        token = _get_token_from_file()
    
    if not token:
        print("Warning: No Hugging Face token found.")
        print("  Options:")
        print("  1. Set environment variable: $env:HF_TOKEN='your_token'")
        print("  2. Create hg.txt file with your token")
        print("  3. Run: huggingface-cli login")
        return False
    
    # Authenticate with the token
    try:
        login(token=token, add_to_git_credential=False)
        user_info = whoami()
        print(f"✓ Authenticated as: {user_info.get('name', 'Unknown')}")
        _authenticated = True
        _hf_token = token  # Store token for explicit use
        
        # Also set as environment variable for transformers to use
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        
        return True
    except Exception as e:
        print(f"✗ Authentication failed: {e}")
        print("  Please check your token is valid at: https://huggingface.co/settings/tokens")
        return False


def load_model(model_name: str = DEFAULT_MODEL_NAME, force_reload: bool = False):
    """
    Load tokenizer and model onto GPU.
    
    Args:
        model_name: Hugging Face model identifier (default: Llama-3.1-8B-Instruct)
        force_reload: If True, reload even if model is already loaded
    
    Returns:
        tuple: (tokenizer, model) objects
    
    Notes:
        - Uses bfloat16 precision for better performance on modern GPUs
        - Automatically uses GPU if available (device_map="auto")
        - Model is loaded in evaluation mode (no gradients)
        - First load may take a few minutes to download model weights
    """
    global _tokenizer, _model, _loaded_model_name
    
    # Skip loading if already loaded (unless force_reload)
    if not force_reload and _model is not None and _loaded_model_name == model_name:
        print(f"Model '{model_name}' already loaded. Skipping reload.")
        return _tokenizer, _model
    
    print(f"Loading model: {model_name}")
    print("This may take a few minutes on first run (downloading weights)...")
    
    # Ensure authentication before loading model
    ensure_authentication()
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Model will run on CPU (very slow).")
        device_map = "cpu"
        torch_dtype = torch.float32
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        device_map = "auto"
        # Use bfloat16 for better performance on Ampere+ GPUs (RTX 3090, A100, etc.)
        torch_dtype = torch.bfloat16
    
    # Load tokenizer
    print("Loading tokenizer...")
    # Explicitly pass token - try both old and new API formats
    tokenizer_kwargs = {"trust_remote_code": True}
    
    # Use token parameter (newer API) - this is the preferred way
    if _hf_token:
        tokenizer_kwargs["token"] = _hf_token
        # Also set in environment as backup
        os.environ["HF_TOKEN"] = _hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token
    
    try:
        _tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    except Exception as e:
        error_msg = str(e)
        if "couldn't connect" in error_msg.lower() or "connection" in error_msg.lower():
            print("\n" + "!" * 70)
            print("CONNECTION ERROR")
            print("!" * 70)
            print("\nNetwork connectivity works, but transformers library can't connect.")
            print("\nPossible solutions:")
            print("1. Check if you've accepted the license at:")
            print(f"   https://huggingface.co/{model_name}")
            print("2. Try using a cached model (if available):")
            print("   python test_model.py --model microsoft/Phi-3-mini-4k-instruct")
            print("3. Check firewall/antivirus settings")
            print("4. Verify your token has access to this model")
            print("!" * 70 + "\n")
        raise
    
    # Set pad token if not set (some models don't have a pad token)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    
    # Load model
    print("Loading model (this may take a while)...")
    
    # Phi-3 models require eager attention implementation to avoid DynamicCache errors
    # This fixes: 'DynamicCache' object has no attribute 'seen_tokens'
    model_kwargs = {
        "dtype": torch_dtype,  # Use 'dtype' instead of deprecated 'torch_dtype'
        "device_map": device_map,
        "trust_remote_code": True,
    }
    
    # Explicitly pass token (newer API) - this is the preferred way
    if _hf_token:
        model_kwargs["token"] = _hf_token
        # Also ensure environment variable is set as backup
        os.environ["HF_TOKEN"] = _hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token
    
    # Use eager attention for Phi-3 models to avoid compatibility issues
    if "phi" in model_name.lower() or "Phi" in model_name:
        model_kwargs["attn_implementation"] = "eager"
        print("Using eager attention implementation for Phi-3 compatibility")
    
    # Optional: use 4-bit quantization if you have <16GB VRAM
    # model_kwargs["load_in_4bit"] = True  # Uncomment if needed
    
    try:
        _model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except Exception as e:
        error_msg = str(e)
        if "couldn't connect" in error_msg.lower() or "connection" in error_msg.lower():
            print("\n" + "!" * 70)
            print("CONNECTION ERROR")
            print("!" * 70)
            print("\nNetwork connectivity works, but transformers library can't connect.")
            print("\nPossible solutions:")
            print("1. Check if you've accepted the license at:")
            print(f"   https://huggingface.co/{model_name}")
            print("2. Try using a cached model (if available):")
            print("   python test_model.py --model microsoft/Phi-3-mini-4k-instruct")
            print("3. Check firewall/antivirus settings")
            print("4. Verify your token has access to this model")
            print("!" * 70 + "\n")
        raise
    
    # Set to evaluation mode
    _model.eval()
    
    _loaded_model_name = model_name
    
    # Print memory usage
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(0) / 1e9
        memory_reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"Model loaded successfully!")
        print(f"GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
    else:
        print("Model loaded successfully (CPU mode)")
    
    return _tokenizer, _model


def generate_text(
    prompt: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    model_name: Optional[str] = None
) -> str:
    """
    Generate text from a prompt using the loaded model.
    
    Args:
        prompt: Input text prompt
        max_new_tokens: Maximum number of new tokens to generate (default: 256)
                       TODO: Adjust based on your needs (higher = longer responses)
        temperature: Sampling temperature (default: 0.7)
                     TODO: Lower (0.1-0.5) for factual tasks, higher (0.7-1.0) for creative tasks
        top_p: Nucleus sampling parameter (default: 0.9)
               TODO: Lower (0.5-0.8) for more focused, higher (0.9-0.95) for more diverse
        model_name: Optional model name (if None, uses already loaded model or default)
    
    Returns:
        Generated text (only the new tokens, not including the prompt)
    
    Example:
        >>> load_model()
        >>> output = generate_text("Explain quantum computing:", max_new_tokens=100)
        >>> print(output)
    """
    global _tokenizer, _model
    
    # Load model if not already loaded
    if _model is None or (model_name is not None and model_name != _loaded_model_name):
        load_model(model_name or DEFAULT_MODEL_NAME)
    
    if _tokenizer is None or _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    
    # Tokenize input
    inputs = _tokenizer(prompt, return_tensors="pt")
    
    # Move inputs to same device as model
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Generate
    # Check if this is a Phi-3 model to handle compatibility issues
    is_phi3 = _loaded_model_name and ("phi" in _loaded_model_name.lower() or "Phi" in _loaded_model_name)
    
    with torch.no_grad():  # No gradients needed for inference
        generate_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,  # Enable sampling
            "pad_token_id": _tokenizer.eos_token_id,
        }
        
        # Phi-3 models have cache compatibility issues - disable cache
        if is_phi3:
            generate_kwargs["use_cache"] = False
        
        outputs = _model.generate(**generate_kwargs)
    
    # Decode only the new tokens (remove the prompt from output)
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    generated_text = _tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return generated_text


def compute_logprob_and_states(
    prompt: str,
    target_answer: str,
    model_name: Optional[str] = None,
    return_all_layers: bool = False,
    return_token_logprobs: bool = False
) -> dict:
    """
    Computes the exact log-probability of a target answer given a prompt using Teacher Forcing,
    and extracts the hidden state representations for the target answer tokens.
    
    Args:
        prompt: The context and query prompt.
        target_answer: The correct/baseline answer to compute likelihood for.
        model_name: Model to use.
    
    Returns:
        dict: containing 'logprob' (float) and 'hidden_states_mean' (dict of layer L2 representations).
    """
    global _tokenizer, _model
    
    if _model is None or (model_name is not None and model_name != _loaded_model_name):
        load_model(model_name or DEFAULT_MODEL_NAME)
        
    if _tokenizer is None or _model is None:
        raise RuntimeError("Model not loaded.")
        
    # Tokenize separately to identify boundaries
    prompt_inputs = _tokenizer(prompt, return_tensors="pt")
    target_inputs = _tokenizer(target_answer, return_tensors="pt", add_special_tokens=False)
    
    input_ids = torch.cat([prompt_inputs.input_ids, target_inputs.input_ids], dim=-1)
    
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
        
    target_len = target_inputs.input_ids.shape[1]
    prompt_len = prompt_inputs.input_ids.shape[1]
    
    with torch.no_grad():
        # Get logits and hidden states
        outputs = _model(input_ids, output_hidden_states=True)
        logits = outputs.logits
        
    # Logits are shifted by 1 relative to input_ids for next-token prediction
    # The first token of target_answer is at prompt_len, predicted by logits at prompt_len - 1
    target_logits = logits[0, prompt_len-1 : prompt_len + target_len - 1, :]
    
    log_probs = torch.nn.functional.log_softmax(target_logits, dim=-1)
    target_token_ids = target_inputs.input_ids[0].cuda() if torch.cuda.is_available() else target_inputs.input_ids[0]
    gathered_log_probs = torch.gather(log_probs, 1, target_token_ids.unsqueeze(-1)).squeeze(-1)
    total_logprob = gathered_log_probs.sum().item()
    
    token_logprobs = None
    tokens_text = None
    if return_token_logprobs:
        token_logprobs = gathered_log_probs.cpu().tolist()
        tokens_text = [_tokenizer.decode([tid]) for tid in target_token_ids]
    
    # Extract hidden states memory-efficiently (early, mid, late layers)
    num_layers = len(outputs.hidden_states)
    if return_all_layers:
        layers_to_extract = list(range(num_layers))
    else:
        layers_to_extract = [0, num_layers // 2, num_layers - 1]
    
    extracted_states = {}
    for layer_idx in layers_to_extract:
        # Extract states corresponding to the TARGET answer
        states = outputs.hidden_states[layer_idx][0, prompt_len:, :].float().cpu().numpy()
        # Mean pool across sequence length to avoid massive data dumps
        mean_state = states.mean(axis=0)
        extracted_states[f"layer_{layer_idx}"] = mean_state.tolist()
        
    result = {
        "logprob": total_logprob,
        "hidden_states_mean": extracted_states
    }
    if return_token_logprobs:
        result["token_logprobs"] = token_logprobs
        result["tokens_text"] = tokens_text
        
    return result


def get_model_info() -> dict:
    if _model is None:
        return {"status": "No model loaded"}
    
    info = {
        "model_name": _loaded_model_name,
        "device": next(_model.parameters()).device,
        "dtype": next(_model.parameters()).dtype,
    }
    
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated(0) / 1e9
        info["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved(0) / 1e9
    
    return info





