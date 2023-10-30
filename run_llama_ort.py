import onnxruntime as ort

sess = ort.InferenceSession("TinyLlama-1.1B-step-50K-105b-onnx/decoder_with_past_model.onnx", providers=["CPUExecutionProvider"])
input_names = list(map(lambda inpt: inpt.name, sess.get_inputs()))
print(input_names)
input_shapes = list(map(lambda inpt: inpt.shape, sess.get_inputs()))
print(input_shapes)

