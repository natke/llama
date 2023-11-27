import os
import argparse
import datetime
import logging
import json
import numpy as np
import onnxruntime
from transformers import LlamaTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

# Perform the one-off intialization for the prediction. The init code is run once when the endpoint is setup.
def init():
    logging.info("Running init() ...")

    global session, model, tokenizer

    ## TODO Do these need to be fixed
    device = "cuda"
    precision = "fp16"
    name = "meta-llama/Llama-2-7b-chat-hf"

    model_name = "rank_0_Llama-2-7b-chat-hf_decoder_merged_model_fp16.onnx"

    # use AZUREML_MODEL_DIR to get your deployed model(s). If multiple models are deployed, 
    # model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), '$MODEL_NAME/$VERSION/$MODEL_FILE_NAME')
    model_dir = os.getenv('AZUREML_MODEL_DIR')

    logging.info("Loading model ...")

    if device == "cuda":
        provider = "CUDAExecutionProvider"
    else:
        provider = "CPUExecutionProvider"

    
    logging.info("Loading tokenizer ...")
    tokenizer = LlamaTokenizer.from_pretrained(f"{name}", use_auth_token=True)
    tokenizer.pad_token = "[PAD]"

    logging.info("Loading model ...")
    model = ORTModelForCausalLM.from_pretrained(
       f'{model_dir}/{name}',
       file_name=f"rank_0_{name.split('/')[1]}_decoder_merged_model_{precision}.onnx",
       use_auth_token=True,
       cache_dir="model_cache",
       provider=provider
    )
    
    
# Run the ONNX model with ONNX Runtime
def run(payload):
    
    data = json.loads(payload)

    logging.info("Running tokenizer ...")
    inputs = tokenizer(data['prompt'], padding=True, return_tensors="pt").to(device)

    logging.info("Running generate ...")
    # Generate
    start_time = datetime.datetime.now()  
    generate_ids = model.generate(**inputs, max_new_tokens=data['new_tokens'], do_sample=True, top_p=0.9)   
    num_tokens = generate_ids.size(dim=1)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0] 
    end_time = datetime.datetime.now()

    logging.info(output)  
    
    seconds = (end_time - start_time).total_seconds()
    logging.info(f"Optimum + ONNX Runtime, {num_tokens}, {new_tokens}, {round(seconds, 2)}, {round(num_tokens / seconds, 1)}, {round(new_tokens / seconds, 1)}")
    
    results = {}
    
    results["output"] = output

    return results


if __name__ == '__main__':
    init()

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

    payload = json.dumps({'prompt': prompt, 'new_tokens': new_tokens, 'precision': precision, 'device': device, 'name': name})

    print(run(payload))


