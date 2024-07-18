import os
import argparse
import datetime
import logging
import json
import numpy as np
import onnxruntime_genai as og
from huggingface_hub import login

# Perform the one-off intialization for the prediction. The init code is run once when the endpoint is setup.
def init():
    logging.info("Running init() ...")

    logging.info(f"{os.getenv('HUGGINGFACE_TOKEN')}")
 
    huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
    login(token=huggingface_token)

    global model, tokenizer, tokenizer_stream, device

    ## TODO Do these need to be fixed
    device = "cuda"
    #precision = os.getenv('PRECISION')
    precision = "fp16"
    #name = os.getenv('MODEL_NAME')
    name = "../../models/meta-llama/Llama-3-8b-int4-cpu-onnx/llama3-8b-int4-cpu"

    # use AZUREML_MODEL_DIR to get your deployed model(s). If multiple models are deployed, 
    # model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), '$MODEL_NAME/$VERSION/$MODEL_FILE_NAME')
    model_dir = os.getenv('AZUREML_MODEL_DIR')


    logging.info("Loading model ...")
    model=og.Model(f"{model_dir}/meta-llama/Llama-3-8b-int4-cpu-onnx/llama3-8b-int4-cpu")

    logging.info("Loading tokenizer ...")
    tokenizer=og.Tokenizer(model)
    tokenizer_stream=tokenizer.create_stream()

    logging.info("Model loaded")
    
# Run the ONNX model with ONNX Runtime
def run(payload):
    
    data = json.loads(payload)

    logging.info("Running tokenizer ...")

    # Keep asking for input prompts in a loop
    input_ids = tokenizer.encode(data['prompt'])

    logging.info("Running generate ...")
    # Generate
    start_time = datetime.datetime.now() 
    max_length = data['max_length'] 

    params=og.GeneratorParams(model)
    params.input_ids = input_ids
    params.set_search_options(max_length=max_length)
     
    generator = og.Generator(model, params)

    output = ""
    new_tokens = []
    while not generator.is_done():
        generator.compute_logits()
        generator.generate_next_token()

        new_token = generator.get_next_tokens()[0]
        new_tokens.append(new_token)
        new_word = tokenizer_stream.decode(new_token) 
        output += new_word

    end_time = datetime.datetime.now()

    logging.info(output)  
    
    seconds = (end_time - start_time).total_seconds()
    logging.info(f"ONNX Runtime generate, {len(new_tokens)}, {new_tokens}, {round(seconds, 2)}, {round(len(new_tokens) / seconds, 1)}")
    
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
    argparser.add_argument('--max_length', type=int, default=256, help='Number of tokens to generate')

    args = argparser.parse_args()

    name = args.name
    prompt = args.prompt
    precision = args.precision
    device = args.device
    max_length = args.max_length

    payload = json.dumps({'prompt': prompt, 'max_length': max_length, 'precision': precision, 'device': device, 'name': name})

    print(run(payload))


