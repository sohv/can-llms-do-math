import json
import os
from typing import List, Dict

def validate_dataset(json_path: str) -> List[Dict]:
    required_fields = ["numbers", "response"]
    with open(json_path, 'r') as f:
        data = json.load(f)
    cleaned = []
    for i, entry in enumerate(data):
        if all(field in entry for field in required_fields):
            cleaned.append(entry)
        else:
            print(f"Entry {i} missing required fields: {entry}")
    print(f"Validated {len(cleaned)} out of {len(data)} entries.")
    return cleaned

def generate_direct_prompt(numbers: List[int]) -> str:
    return f"What is the sum of the following numbers? {', '.join(str(n) for n in numbers)}"

def generate_cot_prompt(numbers: List[int]) -> str:
    return (
        "Let's solve this step by step. "
        f"Add the following numbers one by one: {', '.join(str(n) for n in numbers)}. "
        "Show your reasoning and give the final answer."
    )

def evaluate_llm(dataset: List[Dict], prompt_type: str = "direct", model_call_fn=None, output_path="llm_results.json"):
    results = []
    for entry in dataset:
        numbers = entry["numbers"]
        if prompt_type == "direct":
            prompt = generate_direct_prompt(numbers)
        else:
            prompt = generate_cot_prompt(numbers)
        if model_call_fn:
            model_response = model_call_fn(prompt)
        else:
            model_response = "<model output here>"
        results.append({
            "numbers": numbers,
            "prompt": prompt,
            "model_response": model_response,
            "ground_truth": entry["response"]
        })
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved LLM results to {output_path}")

if __name__ == "__main__":
    dataset_path = os.path.join("data", "additions_r1half-4.json")
    dataset = validate_dataset(dataset_path)
    print("\nExample Direct Prompt:")
    print(generate_direct_prompt(dataset[0]["numbers"]))
    print("\nExample Chain-of-Thought Prompt:")
    print(generate_cot_prompt(dataset[0]["numbers"]))
    print("\nEvaluation pipeline ready. Plug in your LLM API call to run evaluation.") 