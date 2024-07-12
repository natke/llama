import onnxruntime_genai as og

print("Loading model...")

# Generate the model with the following command:
# NOTE: requires ~56GB RAM to run the following command
# python -m onnxruntime.models.builder [-m <HF model name> | -i <path to local PyTorch model] -e cpu -p int4 -o <output path>
#  
# Replace with the path to the model
model=og.Model("../../models/meta-llama/Llama-3-8b-int4-cpu-onnx/llama3-8b-int4-cpu")

print("Model loaded")

tokenizer=og.Tokenizer(model)
tokenizer_stream=tokenizer.create_stream()

# Keep asking for input prompts in an loop
prompt = "Tell me a short joke:"
input_ids = tokenizer.encode(prompt)

params=og.GeneratorParams(model)
params.input_ids = input_ids
params.set_search_options(max_length=20)

generator = og.Generator(model, params)

print(prompt, end=' ', flush=True)

while not generator.is_done():
    generator.compute_logits()
    generator.generate_next_token()

    new_token = generator.get_next_tokens()[0]
    print(tokenizer_stream.decode(new_token), end='', flush=True)

print("\n")


