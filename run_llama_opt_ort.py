import argparse
import datetime
from transformers import LlamaTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

def load_model(name, precision, device):
    return ORTModelForCausalLM.from_pretrained(
       f'models/{name}',
       file_name=f"rank_0_{name.split('/')[1]}_decoder_merged_model_{precision}.onnx",
       use_auth_token=True,
       cache_dir="model_cache",
       provider=provider
    )

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--name', type=str, default='meta-llama/Llama-2-7b-chat-hf', help='Llama model name to export and run')
    argparser.add_argument('--prompt', type=str, default='I like walking my cute dog', help='Prompt to run Llama with')
    argparser.add_argument('--precision', type=str, default='fp16', help='The precision of the model to load')
    argparser.add_argument('--device', type=str, default='cuda', help='Where to run the model')
    argparser.add_argument('--new_tokens', type=int, default=256, help='Number of tokens to generate')

    args = argparser.parse_args()

    name = args.name
    prompt = args.prompt
    precision = args.precision
    device = args.device
    new_tokens = args.new_tokens

    if device == "cuda":
        provider = "CUDAExecutionProvider"
    else:
        provider = "CPUExecutionProvider"

    
    print("Loading tokenizer ...")
    tokenizer = LlamaTokenizer.from_pretrained(f"{name}", use_auth_token=True)
    tokenizer.pad_token = "[PAD]"

    print("Loading model ...")
    model = load_model(name, precision, device)

    print("Running tokenizer ...")
    inputs = tokenizer(prompt, padding=True, return_tensors="pt").to(device)

    print("Running generate ...")
    # Generate
    start_time = datetime.datetime.now()  
    generate_ids = model.generate(**inputs, max_new_tokens=new_tokens, do_sample=True, top_p=0.9)   
    num_tokens = generate_ids.size(dim=1)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
    end_time = datetime.datetime.now()

    print(output)  
    seconds = (end_time - start_time).total_seconds()
    #print(f"Total tokens per second = {round(num_tokens / seconds, 1)} ({num_tokens} in {round(seconds, 1)}s)")
    #print(f"New tokens per second = {round(new_tokens / seconds, 1)} ({new_tokens} in {round(seconds, 2)}s)")
    print(f"Optimum + ONNX Runtime, {num_tokens}, {new_tokens}, {round(seconds, 2)}, {round(num_tokens / seconds, 1)}, {round(new_tokens / seconds, 1)}")
