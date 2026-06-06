#!/usr/bin/env python3
"""LLM inference benchmark on AMD ROCm."""
import torch
import time

def benchmark(model_name="meta-llama/Llama-2-7b-hf", iterations=10):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    inputs = tokenizer("The future of GPU computing is", return_tensors="pt").to(model.device)
    for _ in range(3):
        with torch.no_grad(): model.generate(**inputs, max_new_tokens=10)
    torch.cuda.synchronize()
    start = time.perf_counter()
    tokens = 0
    for _ in range(iterations):
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=100)
        tokens += out.shape[1] - inputs['input_ids'].shape[1]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Tokens/sec: {tokens / elapsed:.2f}")

if __name__ == '__main__': benchmark()
