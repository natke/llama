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


## Run PyTorch model

```bash
python run_llama_pt.py --name <model> --prompt <prompt>
```

where &lt;model&gt; can be:
* `PY007/TinyLlama-1.1B-intermediate-step-480k-1T`
* `PY007/TinyLlama-1.1B-Chat-v0.3`
* `TinyLlama/TinyLlama-1.1B-Chat-v0.6`
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
   python -m onnxruntime.transformers.models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-chat-hf --output models/meta-llama/Llama-2-7b-chat-hf --execution_provider cuda --precision fp16 --use_gqa
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

1. Generate the optimized ONNX model with model builder

   ```bash
   python -m onnxruntime.models.builder [-m <HF model name> | -i <path to local PyTorch model] -e cpu -p int4 -o <output path>
   ```

2. Run the script to generate text with Llama

   ```bash
   cd to the directory with your script and models
   python run_llama_genai_ort.py
   ```


## Deploy and run cloud endpoint in Azure

See https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints?view=azureml-api-2&tabs=azure-cli

### Test scoring script locally

1. Install docker
   
2. Install other dependencies as well as `azureml-inference-server-http`

3. Run `azmlinfsrv --entry_script score.py`

4. In another terminal, send a prompt to the endpoint

   ```bash
   curl --header "Content-Type: application/json" \
        --request POST --data @data.json \
        http://localhost:5001/score
   ```

### Test endpoint and deployment locally

1. Setup

```bash
az extension add --name ml --alow-preview
```

2. Setup the endpoint name, resource group etc
   
   **Linux/Mac**

   ```bash
   export ENDPOINT_NAME=llama
   export AZURE_RESOURCE_GROUP=...
   export AZURE_SUBSCRIPTION=...
   export AZURE_MACHINE_LEARNING_WORKSPACE=...
   export HUGGINGFACE_TOKEN=...
   az login --use-device-code
   az account set --subscription ${AZURE_SUBSCRIPTION}
   az configure --defaults workspace=${AZURE_MACHINE_LEARNING_WORKSPACE} group=${AZURE_RESOURCE_GROUP}
   ```

   **Windows**

   ```cmd
   set ENDPOINT_NAME=llama
   set AZURE_RESOURCE_GROUP=...
   set AZURE_SUBSCRIPTION=...
   set AZURE_MACHINE_LEARNING_WORKSPACE=...
   set HUGGINGFACE_TOKEN=...
   az login --use-device-code
   az account set --subscription %AZURE_SUBSCRIPTION%
   az configure --defaults workspace=%AZURE_MACHINE_LEARNING_WORKSPACE% group=%AZURE_RESOURCE_GROUP%
   ```


3. Create a local endpoint

   ```bash
   az ml online-endpoint create --local -n $ENDPOINT_NAME -f endpoint.yml
   ```

4. Create a local deployment

   Note: a slightly different deployment spec is required for local deployments, as the model is specified as a local path rather than model deployment identifier.

   ```bash
   az ml online-deployment create --local -n blue --endpoint $ENDPOINT_NAME -f deploy-local.yml --set environment_variables.HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}
   ```

5. (Optional) Update the deployment

   ```bash
   az ml online-deployment update --local -n blue --endpoint $ENDPOINT_NAME -f deploy.yml
   ```

### Run the endpoint in Azure

1. Upload the model to Azure

   Model variants and names
   - meta-llama/Llama-2-7b-chat-hf-fp16
   - PY007/TinyLlama-1.1B-Chat-V0.3


   ```bash
   cd models/meta-llama
   az ml model create --name Llama-2-7b-chat-hf-fp16 --path Llama-2-7b-chat-hf
   ```

2. Create the endpoint in Azure

   ```bash
   az ml online-endpoint create -n $ENDPOINT_NAME -f endpoint.yml
   ```

3. Create the deployment in Azure

   ```bash
   az ml online-deployment create -n blue --endpoint $ENDPOINT_NAME -f deploy.yml --set environment_variables.HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}
   ```

4. Allocate traffic to the endpoint

   ```bash
   az ml online-endpoint update --name $ENDPOINT_NAME --traffic "green=100"
   ```

5. Consume the online endpoint

   ```bash
   curl -H "Authorization: Bearer ${ENDPOINT_TOKEN}" --data @data.json https://llama.australiaeast.inference.ml.azure.com/score
   ```

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

### Azure ML errors

    output_buffer = torch.empty(np.prod(output_shape), dtype=torch_type, device=self.device).contiguous()
torch.cuda.OutOfMemoryError: CUDA out of memory.
Tried to allocate 20.00 MiB.
GPU 0 has a total capacty of 15.77 GiB of which 12.88 MiB is free. Process 13953 has 15.76 GiB memory in use.
  Of the allocated memory 480.73 MiB is allocated by PyTorch, and 63.27 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF

  PYTORCH_NO_CUDA_MEMORY_CACHING=1