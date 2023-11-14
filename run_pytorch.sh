bash run_llama_pt_prompt_sequence.sh 0 2> /dev/null | grep 'PyTorch' > pytorch.csv
bash run_llama_pt_prompt_sequence.sh 64 2> /dev/null | grep 'PyTorch' >> pytorch.csv
bash run_llama_pt_prompt_sequence.sh 128 2> /dev/null | grep 'PyTorch' >> pytorch.csv
bash run_llama_pt_prompt_sequence.sh 256 2> /dev/null | grep 'PyTorch' >> pytorch.csv
bash run_llama_pt_prompt_sequence.sh 512 2> /dev/null | grep 'PyTorch' >> pytorch.csv
