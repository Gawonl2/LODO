"""
Test script for LLaMA model wrapper.

This script provides comprehensive testing for the llama_model module,
including basic functionality, parameter variations, and edge cases.

Usage:
    python test_model.py [--model MODEL_NAME]
    
    Examples:
        python test_model.py
        python test_model.py --model microsoft/Phi-3-mini-4k-instruct
        python test_model.py --model Qwen/Qwen2-7B-Instruct
"""

import sys
import argparse
from llama_model import load_model, generate_text, get_model_info, DEFAULT_MODEL_NAME


def check_authentication_error(error_msg: str) -> bool:
    """Check if error is due to authentication issues."""
    auth_keywords = ["gated repo", "401", "authenticated", "access", "login", "restricted"]
    return any(keyword.lower() in str(error_msg).lower() for keyword in auth_keywords)


def print_authentication_help():
    """Print helpful instructions for authentication."""
    print("\n" + "!" * 70)
    print("AUTHENTICATION REQUIRED")
    print("!" * 70)
    print("\nThe model you're trying to use requires Hugging Face authentication.")
    print("\nTo fix this:")
    print("\n1. Login to Hugging Face CLI:")
    print("   huggingface-cli login")
    print("\n2. Or set your token as an environment variable:")
    print("   $env:HF_TOKEN='your_token_here'  # PowerShell")
    print("   export HF_TOKEN='your_token_here'  # Linux/Mac")
    print("\n3. For Meta-Llama models, you also need to:")
    print("   - Visit: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
    print("   - Request access from Meta")
    print("   - Wait for approval")
    print("\n4. Alternative: Use an open model (no authentication needed):")
    print("   python test_model.py --model microsoft/Phi-3-mini-4k-instruct")
    print("   python test_model.py --model Qwen/Qwen2-7B-Instruct")
    print("\n" + "!" * 70)


def test_model_loading(model_name: str = None):
    """Test 1: Model loading functionality."""
    print("\n" + "=" * 70)
    print("TEST 1: Model Loading")
    print("=" * 70)
    
    try:
        tokenizer, model = load_model(model_name or DEFAULT_MODEL_NAME)
        print("✓ Model loaded successfully")
        
        info = get_model_info()
        print("\nModel Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        return True
    except Exception as e:
        error_msg = str(e)
        print(f"✗ Model loading failed: {error_msg}")
        
        if check_authentication_error(error_msg):
            print_authentication_help()
        
        return False


def test_basic_generation(model_name: str = None):
    """Test 2: Basic text generation."""
    print("\n" + "=" * 70)
    print("TEST 2: Basic Text Generation")
    print("=" * 70)
    
    try:
        prompt = "What is artificial intelligence?"
        print(f"\nPrompt: {prompt}")
        print("\nGenerating response...")
        
        output = generate_text(prompt, max_new_tokens=100, model_name=model_name)
        print(f"\n✓ Generation successful")
        print(f"\nResponse:\n{output}")
        
        if len(output.strip()) == 0:
            print("⚠ Warning: Empty response generated")
            return False
        
        return True
    except Exception as e:
        error_msg = str(e)
        print(f"✗ Generation failed: {error_msg}")
        
        if check_authentication_error(error_msg):
            print_authentication_help()
        
        return False


def test_different_prompts(model_name: str = None):
    """Test 3: Different types of prompts."""
    print("\n" + "=" * 70)
    print("TEST 3: Different Prompt Types")
    print("=" * 70)
    
    test_cases = [
        {
            "name": "Question Answering",
            "prompt": "Explain the difference between machine learning and deep learning.",
            "max_tokens": 150
        },
        {
            "name": "Creative Writing",
            "prompt": "Write a haiku about technology.",
            "max_tokens": 50
        },
        {
            "name": "Code Explanation",
            "prompt": "What does this Python code do: def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
            "max_tokens": 100
        }
    ]
    
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] {test_case['name']}")
        print(f"Prompt: {test_case['prompt']}")
        try:
            output = generate_text(
                prompt=test_case['prompt'],
                max_new_tokens=test_case['max_tokens'],
                model_name=model_name
            )
            print(f"✓ Success")
            print(f"Response: {output[:200]}..." if len(output) > 200 else f"Response: {output}")
            results.append(True)
        except Exception as e:
            error_msg = str(e)
            print(f"✗ Failed: {error_msg}")
            if check_authentication_error(error_msg) and i == 1:
                print_authentication_help()
            results.append(False)
    
    return all(results)


def test_parameter_variations(model_name: str = None):
    """Test 4: Different generation parameters."""
    print("\n" + "=" * 70)
    print("TEST 4: Parameter Variations")
    print("=" * 70)
    
    base_prompt = "Describe the future of AI in one sentence."
    
    test_configs = [
        {
            "name": "Low Temperature (Deterministic)",
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 50
        },
        {
            "name": "High Temperature (Creative)",
            "temperature": 0.9,
            "top_p": 0.95,
            "max_tokens": 50
        },
        {
            "name": "Medium Temperature (Balanced)",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 50
        }
    ]
    
    results = []
    for i, config in enumerate(test_configs, 1):
        print(f"\n[{i}/{len(test_configs)}] {config['name']}")
        print(f"Parameters: temp={config['temperature']}, top_p={config['top_p']}")
        try:
            output = generate_text(
                prompt=base_prompt,
                max_new_tokens=config['max_tokens'],
                temperature=config['temperature'],
                top_p=config['top_p'],
                model_name=model_name
            )
            print(f"✓ Success")
            print(f"Response: {output}")
            results.append(True)
        except Exception as e:
            error_msg = str(e)
            print(f"✗ Failed: {error_msg}")
            if check_authentication_error(error_msg) and i == 1:
                print_authentication_help()
            results.append(False)
    
    return all(results)


def test_edge_cases(model_name: str = None):
    """Test 5: Edge cases and error handling."""
    print("\n" + "=" * 70)
    print("TEST 5: Edge Cases")
    print("=" * 70)
    
    edge_cases = [
        {
            "name": "Empty prompt",
            "prompt": "",
            "should_fail": False  # Model might handle empty prompts
        },
        {
            "name": "Very short prompt",
            "prompt": "Hi",
            "should_fail": False
        },
        {
            "name": "Very long prompt",
            "prompt": "Tell me about " + "machine learning " * 50,
            "should_fail": False
        },
        {
            "name": "Special characters",
            "prompt": "What is @#$%^&*()?",
            "should_fail": False
        }
    ]
    
    results = []
    for i, case in enumerate(edge_cases, 1):
        print(f"\n[{i}/{len(edge_cases)}] {case['name']}")
        print(f"Prompt: {case['prompt'][:50]}..." if len(case['prompt']) > 50 else f"Prompt: {case['prompt']}")
        try:
            output = generate_text(case['prompt'], max_new_tokens=30, model_name=model_name)
            if case['should_fail']:
                print(f"⚠ Unexpected success (expected failure)")
                results.append(False)
            else:
                print(f"✓ Handled successfully")
                print(f"Response: {output[:100]}..." if len(output) > 100 else f"Response: {output}")
                results.append(True)
        except Exception as e:
            error_msg = str(e)
            if case['should_fail']:
                print(f"✓ Failed as expected: {error_msg}")
                results.append(True)
            else:
                print(f"✗ Unexpected failure: {error_msg}")
                if check_authentication_error(error_msg) and i == 1:
                    print_authentication_help()
                results.append(False)
    
    return all(results)


def test_multiple_generations(model_name: str = None):
    """Test 6: Multiple sequential generations (testing caching)."""
    print("\n" + "=" * 70)
    print("TEST 6: Multiple Sequential Generations")
    print("=" * 70)
    
    prompts = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?"
    ]
    
    print("\nTesting that model can handle multiple generations without reloading...")
    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Generating for: {prompt}")
        try:
            output = generate_text(prompt, max_new_tokens=50, model_name=model_name)
            print(f"✓ Success")
            print(f"Response: {output[:100]}..." if len(output) > 100 else f"Response: {output}")
            results.append(True)
        except Exception as e:
            error_msg = str(e)
            print(f"✗ Failed: {error_msg}")
            if check_authentication_error(error_msg) and i == 1:
                print_authentication_help()
            results.append(False)
    
    return all(results)


def run_all_tests(model_name: str = None):
    """Run all test functions and report results."""
    print("\n" + "=" * 70)
    print("LLaMA MODEL TEST SUITE")
    print("=" * 70)
    
    if model_name:
        print(f"\nUsing model: {model_name}")
    else:
        print(f"\nUsing default model: {DEFAULT_MODEL_NAME}")
    
    tests = [
        ("Model Loading", lambda: test_model_loading(model_name)),
        ("Basic Generation", lambda: test_basic_generation(model_name)),
        ("Different Prompts", lambda: test_different_prompts(model_name)),
        ("Parameter Variations", lambda: test_parameter_variations(model_name)),
        ("Edge Cases", lambda: test_edge_cases(model_name)),
        ("Multiple Generations", lambda: test_multiple_generations(model_name)),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test LLaMA model wrapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_model.py
  python test_model.py --model microsoft/Phi-3-mini-4k-instruct
  python test_model.py --model Qwen/Qwen2-7B-Instruct
  
Open models (no authentication needed):
  - microsoft/Phi-3-mini-4k-instruct (~3.8B, ~8GB VRAM)
  - Qwen/Qwen2-7B-Instruct (~7B, ~14GB VRAM, Apache 2.0)
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use (default: from llama_model.py)"
    )
    
    args = parser.parse_args()
    
    try:
        exit_code = run_all_tests(model_name=args.model)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

