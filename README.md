# Llama sample README

This sample shows you how to run the Llama model with PyTorch, HuggingFace and ONNX Runtime.


## Dependencies

* protobuf==3.20.3
* onnx
* onnxruntime (>= 1.16.2)
* torch (>= 2.2.0)
* transformers
* optimum (>= 1.14.0)

## Get access to the Meta model(s)

1. Sign up to get access to the Llama model on HuggingFace (e.g. https://huggingface.co/meta-llama/Llama-2-7b-hf-chat)


## Notes on model variants and execution targets:
* Tiny Llama requires GQA
* GQA only works with fp16 CUDA

## Run PyTorch model

```bash
python run_llama_pt.py --name <model> --prompt <prompt>
```

where &lt;model&gt; can be:
* `PY007/TinyLlama-1.1B-intermediate-step-480k-1T`
* `PY007/TinyLlama-1.1B-Chat-v0.3`
* (default)`meta-llama/Llama-2-7b-hf` if you have access to the gated meta model

## Run Optimum ONNX model

```bash
python run_llama_opt_onnx.py
```

Same options.

## Run ONNX model optimized with ONNX Runtime using Optimum

Export the model with ONNX Runtime

Note: this step requires 54GB of memory

1. Install the latest version of torch

   ```bash 
   pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu118
   ```


2. Install onnxruntime

  ```bash
  pip install onnxruntime-gpu
  ```

3. Export the model

   Note: you cannot have a local folder that is same as the string passed to the `-m` argument, as HuggingFace will look for the model in this folder rather than downloading it and will error if it is not there.

   ```bash
   huggingface-cli login --token <token>
   python -m onnxruntime.transformers.models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-chat-hf --output models/meta-llama/Llama-2-7b-chat-hf --execution_provider cuda --precision fp16
   ```


4. Download the `config.json` and the `generate_config.json` files from HuggingFace and add them to the above folder
    ```bash
    curl -H "Authorization: Bearer <token>" https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/raw/main/config.json > config.json
    curl -H "Authorization: Bearer <token>" https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/raw/main/generation_config.json > generate_config.json
    ```

5. Run the model

   ```bash
   python run_llama_opt_ort.py
   ```

   ```
   usage: run_llama_opt_ort.py [-h] [--name NAME] [--prompt PROMPT] [--precision PRECISION] [--device DEVICE]

   optional arguments:
    -h, --help             show this help message and exit
    --name NAME            Llama model name to export and run
    --prompt PROMPT        Prompt to run Llama with
    --precision PRECISION  The precision of the model to load
    --device DEVICE        Where to run the model
   ```

## Run optimized ONNX model with ONNX Runtime GenAI

Assumes you have CUDA and cmake installed.

1. Clone onnxruntime-genai repo (Temporaty until there is a release package)

   ```bash
   git clone https://github.com/microsoft/onnxruntime-genai.git
   cd onnxruntime-genai
   ```

2. Create a conda env
   ```bash
   conda create -n genai Python=3.9
   ```

3. Install onnxruntime

   ```
   mkdir -p ort
   cd ort
   wget https://raw.githubusercontent.com/microsoft/onnxruntime/v1.16.2/include/onnxruntime/core/session/onnxruntime_c_api.h
   wget https://raw.githubusercontent.com/microsoft/onnxruntime/v1.16.2/include/onnxruntime/core/session/onnxruntime_cxx_api.h
   wget https://raw.githubusercontent.com/microsoft/onnxruntime/v1.16.2/include/onnxruntime/core/session/onnxruntime_cxx_inline.h

   wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.2/onnxruntime-linux-x64-gpu-1.16.2.tgz
   tar xvzf onnxruntime-linux-x64-gpu-1.16.2.tgz
   cp onnxruntime-linux-x64-gpu-1.16.2/lib/libonnxruntime*.so* .
   ```


4. Build onnxruntime-genai

   ```bash
   # Change back into root directory of onnxruntime-genai
   cd ..
   bash build.sh
   ```

5. Set python path so onnxruntime-genai lib can be found by Python (temporary)

   ```bash
   export PYTHONPATH=`pwd`/build
   ```

6. Run the script to generate text with Llama

   ```bash
   cd to the directory with your script and models
   python run_llama_genai_ort.py
   ```

33554432
90177536

## Optional steps

### Build ONNX Runtime from source

```bash
cd onnxruntime
build.sh --config RelWithDebInfo --build_shared_lib --build_wheel --skip_tests --parallel --skip_submodule_sync --use_cuda
```

### Upload the model to Azure blob storage

```bash
az login --use-device-code
az account set --subscription <subscription>
az storage blob upload -f models/meta-llama/Llama-2-7b-chat-hf/rank_0_Llama-2-7b-chat-hf_decoder_merged_model_fp16.onnx --container-name  models --account-name  nakershadevstorage 
az storage blob upload -f models/meta-llama/Llama-2-7b-chat-hf/Llama-2-7b-chat-hf_decoder_merged_model_fp16.onnx.data --container-name  models --account-name  nakershadevstorage 
```