
import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_client import create_llm_client
from hbbops.run_hbbops import extract_answer, GSM8KEvaluator, Prompt

def debug_run():
    print("Initializing LLM...")
    llm_client = create_llm_client(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        backend="vllm",
        device="auto"
    )
    
    evaluator = GSM8KEvaluator(llm_client, debug=True)
    
    # Create a dummy prompt
    instruction = "Solve the following math problem step by step."
    exemplar = "Q: 1+1\nA: 2"
    prompt = Prompt(instruction, exemplar, 0, 0)
    
    # Create dummy validation examples
    val_data = [
        {
            "question": "Janet has 3 apples. She buys 2 more. How many apples does she have?",
            "answer": "She has 3 + 2 = 5 apples. #### 5"
        },
        {
            "question": "John has 10 apples. He eats 4. How many apples does he have left?",
            "answer": "He has 10 - 4 = 6 apples. #### 6"
        }
    ]
    
    print("\nRunning evaluation...")
    error = evaluator(prompt, val_data)
    print(f"\nError rate: {error}")

if __name__ == "__main__":
    debug_run()
