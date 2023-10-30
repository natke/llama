import argparse
import datetime
from transformers import LlamaForCausalLM, LlamaTokenizer

# Name is model name e.g. PY007/TinyLlama-1.1B-step-50K-105b
argparser = argparse.ArgumentParser()
argparser.add_argument('--name', type=str, default='meta-llama/Llama-2-7b-hf', help='Llama model name to run')
argparser.add_argument('--prompt', type=str, default='I like walking my cute dog', help='Prompt to run Llama with')

args = argparser.parse_args()
name = args.name
prompt = args.prompt

tokenizer = LlamaTokenizer.from_pretrained(f"{name}", cache_dir="__cache_dir")
model = LlamaForCausalLM.from_pretrained(f"{name}", cache_dir="__cache_dir")

inputs = tokenizer(prompt, return_tensors="pt")

# Generate
start_time = datetime.datetime.now()
generate_ids = model.generate(inputs.input_ids, max_new_tokens=200, do_sample=True, top_p=0.9)
num_tokens = generate_ids.size(dim=1)
output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
end_time = datetime.datetime.now()

print(output)
seconds = (end_time - start_time).total_seconds()
print(f"Tokens per second = {round(num_tokens / seconds, 1)} ({num_tokens} in {round(seconds, 1)})")
