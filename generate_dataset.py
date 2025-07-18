import ollama
import json
import random

def generate_problem():
    return [random.randint(10**4, 10**7-1) for _ in range(6)]

def get_model_response(problem):
    numbers_str = '+'.join(map(str, problem))
    prompt = f"""{numbers_str}
    Format your final answer as <answer>1234567</answer>"""

    response = ollama.chat(
        model='deepseek-r1:1.5b',
        messages=[{
            'role': 'user',
            'content': prompt
        }]
    )
    
    return response['message']['content']

dataset = []
mistake_count = 0
total_count = 0


for _ in range(500):  # Number of problems
    total_count += 1
    numbers = generate_problem()
    response = get_model_response(numbers)
    
    # Extract thought and generated answer
    # thought = extract_thought(response)
    # generated_answer = extract_answer(response)
    
    # # Calculate actual answer
    # actual_answer = sum(numbers)
    
    # Check if there's a mistake
    # is_mistake = False
    # if generated_answer is None or generated_answer != actual_answer:
    #     is_mistake = True
    #     mistake_count += 1
    
    dataset.append({
        "numbers": numbers,
        "response": response,
        # "thought": thought,
        # "generated_answer": generated_answer,
        # "actual_answer": actual_answer,
        # "mistake": is_mistake
    })

with open('additions_r1half-4.json', 'w') as f:
    json.dump(dataset, f, indent=2)
print(f"Total count: {total_count}, Mistakes: {mistake_count}, Accuracy: {1 - mistake_count / total_count:.2%}")