# Llama sample README

## Dependencies

* onnx
* onnxruntime (>= 1.17.0)
* optimum (>= 1.14.0)
* torch (>= 2.2.0)
* transformers
* protobuf==3.20.3

## Download the Meta model(s)

1. Sign up to get access to the Llama model on HuggingFace (e.g. https://huggingface.co/meta-llama/Llama-2-7b-hf)
2. Download the weights using the script that is emailed from meta
3. Change the name of the folder to `meta-llama/Llama-2-7b-hf`
4. Download the `config.json` and the `generate_config.json` files from HuggingFace and add them to the above folder

## A note on Llama prompting




## Run PyTorch model

```bash
python run_llama_pt.py --name <model> --prompt <prompt>
```

where <model> can be:
* `PY007/TinyLlama-1.1B-intermediate-step-480k-1T`
* (default)`meta-llama/Llama-2-7b-hf` if you downloaded the meta weights

## Run Optimum ONNX model
```bash
EXPORT TRANSFORMERS_CACHE=__cache_dir
```

```bash
python run_llama_opt_onnx.py
```

Same options.

## Run ONNX model optimized with ONNX Runtime

Export the model with ONNX Runtime

Note: this step requires 54GB of memory

1. Install the latest version of torch

   ```bash 
   pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu118
   ```


2. Install the nightly build of onnxruntime

  ```bash
  
  ```

3. Export the model

   ```bash
   git clone <onnxruntime>
   cd onnxruntime/onnxruntime/python/tools/transformers/
   python -m models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output meta-llama/Llama-2-7b-hf-onnx
   cp meta-llama/Llama-2-7b-hf-onnx/* meta-llama/Llama-2-7b-hf
   ```

4. Run the model

```bash
python run_llama_opt_ort.py
```

Same options.

