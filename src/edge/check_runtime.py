import onnxruntime as ort
import numpy as np
import time

import tflite_runtime.interpreter as tflite
import pickle

with open('pretrained/metadata_buffer_names.pkl', 'rb') as f:
    buffer_names = pickle.load(f)

with open('pretrained/metadata_state_buf.pkl', 'rb') as f:
    state_buf = pickle.load(f)

RUNS = 100
tflite_path = 'pretrained/tflite_test/model.tflite'
onnx_path = 'pretrained/ONNX/model.onnx'

sess_opt = ort.SessionOptions()
sess_opt.intra_op_num_threads=4
ort_sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'], sess_options=sess_opt)

# Load TFLite
interpreter = tflite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_names_order = []
for i in range(len(input_details)):
    input_names_order.append(input_details[i]['name'])

output_names_order = []
for i in range(len(output_details)):
    output_names_order.append(output_details[i]['name'])

def get_input(name):
    idx = input_names_order.index(name)
    return input_details[idx]

def get_output(name):
    idx = output_names_order.index(name)
    return output_details[idx]

print('Input Names', input_names_order)
print('output Names', output_names_order)
onnx_input_dict = {'mixture':np.ones((1, 2, 192), dtype=np.float32), 'embedding':np.ones((1, 1, 256), dtype=np.float32)}
input_shape = (1, 1, 192, 2)

quant_type = np.float32
input_data = np.ones(input_shape, dtype=quant_type)

interpreter.set_tensor(get_input("infer_mixture:0")["index"], input_data)

for name in buffer_names:
    zeros = np.zeros(state_buf[name].shape, dtype=quant_type)
    onnx_input_dict[name.replace('__', '::')] = zeros.copy()
    interpreter.set_tensor(get_input(f"infer_{name}:0")["index"], zeros)

t1 = time.time()
for i in range(RUNS):
    ort_sess.run(None, onnx_input_dict)
t2 = time.time()
print('ONNX', (t2 - t1) / RUNS * 1000, 'ms')

t1 = time.time()
for i in range(RUNS):
    interpreter.invoke()
t2 = time.time()
print('TFLite', (t2 - t1) / RUNS * 1000, 'ms')