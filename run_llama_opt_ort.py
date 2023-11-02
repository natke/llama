import argparse
import datetime
from transformers import LlamaTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

def load_model(name):
    return ORTModelForCausalLM.from_pretrained(
       f'models/{name}',
       file_name=f"{name.split('/')[1]}_decoder_merged_model_fp16.onnx",
       use_auth_token=True,
       cache_dir="__cache_dir",
       provider="CUDAExecutionProvider"
    )

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--name', type=str, default='meta-llama/Llama-2-7b-chat-hf', help='Llama model name to export and run')
    argparser.add_argument('--prompt', type=str, default='I like walking my cute dog', help='Prompt to run Llama with')

    args = argparser.parse_args()

    name = args.name
    prompt = args.prompt


    
    print("Loading tokenizer ...")
    tokenizer = LlamaTokenizer.from_pretrained(f"{name}", use_auth_token=True)

    print("Loading model ...")
    model = load_model(name)

    print("Running tokenizer ...")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

    print("Running generate ...")
    # Generate
    start_time = datetime.datetime.now()  
    generate_ids = model.generate(inputs.input_ids, max_new_tokens=200, do_sample=True, top_p=0.9)   
    num_tokens = generate_ids.size(dim=1)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
    end_time = datetime.datetime.now()

    print(output)  
    seconds = (end_time - start_time).total_seconds()
    print(f"Tokens per second = {round(num_tokens / seconds, 1)} ({num_tokens} in {round(seconds, 1)})")
