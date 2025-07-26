import os
import json
import ollama

def ollama_deepseek_call(prompt: str) -> str:
    """Ollama model call for compatibility."""
    response = ollama.chat(
        model='deepseek-r1:1.5b',
        messages=[{
            'role': 'user',
            'content': prompt
        }]
    )
    return response['message']['content']

# Try DSPy import and setup
try:
    import dspy
    from math_llm_pipeline import validate_dataset, evaluate_llm_dspy, evaluate_llm
    
    class SimpleLM:
        """Simplified LM wrapper for DSPy compatibility"""
        def __init__(self, model_fn):
            self.model_fn = model_fn
            self.kwargs = {}
        
        def __call__(self, prompt, **kwargs):
            return [self.model_fn(prompt)]
        
        def generate(self, prompt, **kwargs):
            return [self.model_fn(prompt)]
    
    DSPY_AVAILABLE = True
    print("DSPy available - will run DSPy evaluations")
    
except ImportError as e:
    from math_llm_pipeline import validate_dataset, evaluate_llm
    DSPY_AVAILABLE = False
    print(f"DSPy not available ({e}) - running legacy evaluation only")

if __name__ == "__main__":
    dataset_path = os.path.join("data", "additions_r1half-4.json")
    dataset = validate_dataset(dataset_path)
    
    if DSPY_AVAILABLE:
        print("\n=== Running DSPy Evaluations ===")
        
        # Setup simple LM wrapper
        simple_lm = SimpleLM(ollama_deepseek_call)
        
        print("1. DSPy Direct Prompting...")
        try:
            solver_direct, accuracy_direct = evaluate_llm_dspy(
                dataset, 
                solver_type="direct", 
                lm=simple_lm, 
                optimize=False,
                output_path="results/dspy_direct_results.json"
            )
            print(f"   Direct accuracy: {accuracy_direct:.2%}")
        except Exception as e:
            print(f"   Error in direct evaluation: {e}")
        
        print("\n2. DSPy Chain-of-Thought...")
        try:
            solver_cot, accuracy_cot = evaluate_llm_dspy(
                dataset, 
                solver_type="cot", 
                lm=simple_lm, 
                optimize=False,
                output_path="results/dspy_cot_results.json"
            )
            print(f"   CoT accuracy: {accuracy_cot:.2%}")
        except Exception as e:
            print(f"   Error in CoT evaluation: {e}")
        
        print("\n3. DSPy Optimized (if dataset > 5 examples)...")
        try:
            solver_optimized, accuracy_optimized = evaluate_llm_dspy(
                dataset[:20] if len(dataset) > 20 else dataset,  # Limit for faster optimization
                solver_type="cot", 
                lm=simple_lm, 
                optimize=True,
                output_path="results/dspy_optimized_results.json"
            )
            print(f"   Optimized accuracy: {accuracy_optimized:.2%}")
        except Exception as e:
            print(f"   Error in optimized evaluation: {e}")
    
    print("\n=== Running Legacy Evaluations ===")
    
    print("1. Legacy Direct Prompting...")
    evaluate_llm(dataset, prompt_type="direct", model_call_fn=ollama_deepseek_call, output_path="results/legacy_direct_results.json")
    
    print("2. Legacy Chain-of-Thought...")
    evaluate_llm(dataset, prompt_type="cot", model_call_fn=ollama_deepseek_call, output_path="results/legacy_cot_results.json")
    
    print("\nEvaluation complete. Results saved to results/ directory.") 