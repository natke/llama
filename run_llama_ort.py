import onnxruntime as ort

sess = ort.InferenceSession("models/meta-llama/Llama-2-7b-chat-hf/rank_0_Llama-2-7b-chat-hf_decoder_merged_model_fp16.onnx", providers=["CUDAExecutionProvider"])
input_names = list(map(lambda inpt: inpt.name, sess.get_inputs()))
print(input_names)
input_shapes = list(map(lambda inpt: inpt.shape, sess.get_inputs()))
print(input_shapes)

