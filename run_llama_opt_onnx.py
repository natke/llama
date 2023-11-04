import argparse
import datetime
from transformers import LlamaTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

# Name is model name e.g. PY007/TinyLlama-1.1B-intermediate-step-480k-1T
argparser = argparse.ArgumentParser()
argparser.add_argument('--name', type=str, default='meta-llama/Llama-2-7b-hf', help='Llama model name to export and run')
argparser.add_argument('--prompt', type=str, default='I like walking my cute dog', help='Prompt to run Llama with')
argparser.add_argument('--device', type=str, default='cuda:0', help='Device to run on')

args = argparser.parse_args()

name = args.name
prompt = args.prompt
device = args.device

tokenizer = LlamaTokenizer.from_pretrained(f"{name}", cache_dir="__cache_dir")
model = ORTModelForCausalLM.from_pretrained(f"{name}", export=True, cache_dir="__cache_dir").half().to(device)

# This step is optional, unless you want to save the exported ONNX model to disk
#model.save_pretrained(f"models/{name}-onnx", cache_dir="__cache_dir")

inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate
start_time = datetime.datetime.now()
generate_ids = model.generate(inputs.input_ids, max_new_tokens=200, do_sample=True, top_p=0.9)
num_tokens = generate_ids.size(dim=1)
output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
end_time = datetime.datetime.now()

print(output)
seconds = (end_time - start_time).total_seconds()
print(f"Tokens per second = {round(num_tokens / seconds, 1)} ({num_tokens} in {round(seconds, 1)})")
