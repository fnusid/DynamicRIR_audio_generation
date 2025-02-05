import torch
import numpy as np
import random
import os

import onnx
import onnxruntime as ort

from torch._export import capture_pre_autograd_graph
from torch.export import export, ExportedProgram
# from executorch.exir import ExecutorchBackendConfig, ExecutorchProgramManager
# from executorch.exir import EdgeProgramManager, to_edge

from src.utils import load_net, load_pretrained
from src.edge.edge_utils import flatten_state_buffers, unflatten_state_buffers

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# ONNX Opset to use
opset = 17

# Number of batches
B = 1

# Whether to use model output from simplify_onnx or not
simplify_onnx = True

quantize = False
# quantize = True

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

current_labels = []

device = 'cpu'
# mdl, params = load_net('configs/waveformer.json', return_params=True)
# mdl, params = load_net('configs/tfgridnet_orangepi.json', return_params=True)
# mdl, params = load_net('runs/semhearing_tfgridnet_orangepi_from_baked_with_motion_with_aug/config.json', return_params=True)
# mdl, params = load_pretrained('runs/semhearing_tfgridnet_orangepi_from_baked_with_motion_with_aug', return_params=True, use_last=True)
# mdl, params = load_pretrained('runs/semhearing_tfgridnet_smaller', return_params=True, use_last=False)
# mdl, params = load_pretrained('runs/semhearing_tfgridnet_large_short', return_params=True, use_last=False)
# mdl, params = load_pretrained('runs/semhearing_tfgridnet_large_short_with_asserts', return_params=True, use_last=False)
# mdl, params = load_pretrained('runs/semhearing_tfgridnet_smaller_short_samples', return_params=True, use_last=False)
# mdl, params = load_pretrained('runs/semhearing_LONG_SAMPLES', return_params=True, use_last=False)
# mdl, params = load_pretrained('runs/semhearing_LARGE_SAMPLES_CORRECT', return_params=True, use_last=True)
mdl, params = load_net('configs/tsh_demo.json', return_params=True)

mdl = mdl.tsh_mdl.tfgridnet
mdl.eval()

total_params = sum(p.numel() for p in mdl.parameters())# if p.requires_grad)
print('Number of parameters:', total_params / 1e6)

params = params['pl_module_args']['tsh_model_params']

C = params['num_ch']
# PAD_FRONT = params['lookahead_samples']
# PAD_BACK = params['lookback_samples']
# CHUNK_SIZE = params['chunk_size']
PAD_FRONT = params['stft_pad_size']
PAD_BACK = params['stft_back_pad']
CHUNK_SIZE = params['stft_chunk_size']
D = params['spk_emb_dim']
L = PAD_BACK + (CHUNK_SIZE) + PAD_FRONT

F = L + 2
T = 1
X = torch.randn(B, C, F, T) * 1e1
EMB = torch.randn(B, 1, D) * 1e1

# Model Wrapper
class MyModel(torch.nn.Module):
    def __init__(self, mdl, state_buffer_names) -> None:
        super().__init__()

        self.model = mdl
        self.order = state_buffer_names
        
        # print(self.order)
    
    def forward(self, mix, embedding, *buffers) -> torch.Tensor:
        state_dict = unflatten_state_buffers(self.order, buffers)
        
        outputs, next_state = self.model(mix, embedding, input_state=state_dict)
        
        state_names, next_states = flatten_state_buffers(next_state)

        return outputs, *next_states

def initialize_state_buffers(mdl):
    # Initialize initial state
    init_state = mdl.init_buffers(1, X.device)

    # Get flattened buffers
    buffer_names, buffers = flatten_state_buffers(init_state)

    return buffer_names, buffers

buffer_names, buffers = initialize_state_buffers(mdl)
model = MyModel(mdl, buffer_names)
model.eval()


torch_jit_dir = 'pretrained/TorchJIT'
onnx_dir = 'pretrained/ONNX'
executorch_dir = 'pretrained/ExecuTorch'

torch_jit_path = os.path.join(torch_jit_dir, 'model.pt')
onnx_path = os.path.join(onnx_dir, 'model.onnx')

os.makedirs(torch_jit_dir, exist_ok=True)
os.makedirs(onnx_dir, exist_ok=True)
with torch.no_grad():
    # Create a traced model
    traced_model = torch.jit.trace(model, (X, EMB, *buffers))
    torch.jit.save(traced_model, torch_jit_path)
    
    buffer_names, buffers = initialize_state_buffers(mdl)

    inames = ['mixture', 'embedding'] + buffer_names
    onames = ['filtered_output'] + [f'out::{name}' for name in buffer_names]

    # print(inames)
    # print("EXPECTED OUTPUTS", len(onames))

    # Create ONNX model
    torch.onnx.export(model,
                    (X, EMB, *buffers),
                    onnx_path,
                    export_params=True,
                    input_names = inames,
                    output_names = onames,
                    opset_version=opset)
    
    # # Create executorch model
    # pre_autograd_aten_dialect = capture_pre_autograd_graph(model, (X, EMB, *buffers))
    # # print("Pre-Autograd ATen Dialect Graph")
    # # print(pre_autograd_aten_dialect)

    # aten_dialect: ExportedProgram = export(pre_autograd_aten_dialect, (X, EMB, *buffers))

    # # Convert to edge dialect
    # edge_program: EdgeProgramManager = to_edge(aten_dialect)

    # # Create edge program
    # executorch_program = ExecutorchProgramManager = edge_program.to_executorch(
    #     ExecutorchBackendConfig(
    #         passes=[],  # User-defined passes
    #     )
    # )

    # os.makedirs(executorch_dir, exist_ok=True)
    # with open(os.path.join(executorch_dir, "model.pte"), "wb") as file:
    #     file.write(executorch_program.buffer)

print("[INFO] Converted to onnx!")

if simplify_onnx:
    from onnxsim import simplify
    print("Simplifying model")
    
    onnx.checker.check_model(onnx_path)
    model, check = simplify(onnx_path)
    assert check, "Simplified ONNX model could not be validated"
    
    onnx.save_model(model, onnx_path)

if quantize:
    from onnxruntime.quantization.quantize import quantize_dynamic, QuantFormat, QuantType
    from onnxruntime.quantization.shape_inference import quant_pre_process

    # os.system(f"python -m onnxruntime.quantization.preprocess --input {onnx_path} --output {onnx_path}")
    print("Preparing model for quantization")
    quant_pre_process(onnx_path, onnx_path)
    
    print("Quantizing")
    # quantize_dynamic(onnx_path, onnx_path, op_types_to_quantize=['LSTM'])
    quantize_dynamic(onnx_path, onnx_path, weight_type=QuantType.QUInt8)
    print("Done")

sess_options = ort.SessionOptions()
# sess_options.enable_profiling = True

ort_sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'], sess_options=sess_options)

import time
RUNS = 1000

print("[PyTorch]")
with torch.no_grad():
    t1 = time.time()
    
    for i in range(RUNS):
        gt_output = \
            traced_model(X, EMB, *buffers)[0]
    
    t2 = time.time()
    
    pt_time = t2 - t1

mixed = X.numpy()

inputs = dict(mixture=X.detach().numpy(), embedding=EMB.detach().numpy())
buffer_names, buffers = initialize_state_buffers(mdl)

for name, buf in zip(buffer_names, buffers):
    inputs[name] = buf.detach().numpy()

print("[ONNX]")
t1 = time.time()
for i in range(RUNS):
    output_list = ort_sess.run(None, inputs)
t2 = time.time()

onnx_time = t2 - t1

output = output_list[0]
print(output[0, 0, :20])
print(gt_output.numpy()[0, 0, :20])
print((output - gt_output.numpy())[0, 0, :20])
print(np.allclose(output, gt_output, 1e-4))

print("PT TIMES:", pt_time / RUNS)
print("ONNX TIMES:", onnx_time / RUNS)


# quit()

os.makedirs('pretrained/test_data/replication_test', exist_ok=True)
# Save inputs
input_names = inames
with open('pretrained/test_data/replication_test/input_names.txt', 'w') as f:
    f.write('\n'.join(input_names))

for i in range(len(input_names)):
    fname = os.path.join('pretrained', 'test_data/replication_test', input_names[i] + '.npy')
    np.save(fname, inputs[input_names[i]])

# Save outputs
output_names = onames
with open('pretrained/test_data/replication_test/output_names.txt', 'w') as f:
    f.write('\n'.join(output_names))

for i in range(len(output_names)):
    fname = os.path.join('pretrained', 'test_data/replication_test', output_names[i] + '.npy')
    np.save(fname, output_list[i])

# Save example dataset to run ONNX perftest
print("Creating datasets to run perftest")
import src.edge.ort_test_dir_utils as ort_test_dir_utils
model_path = onnx_path

ort_test_dir_utils.create_test_dir(model_path, '.', 'onnx_perftest_dataset', name_input_map=inputs)
print("Done")

# # END TO END STREAMING MODEL TEST
# print("Creating arrays to run end-to-end streaming test")
# X = torch.randn(B, C, CHUNK_SIZE * 15 + PAD_FRONT) * 1e1

# from src.edge.causal_infer import ModelWrapper, streaming_inference

# model = ModelWrapper(mdl)
# model.eval()

# model_stream = ModelWrapper(mdl)
# model_stream.eval()

# with torch.no_grad():
#     output_full = model.feed(X, embedding=EMB, pad=False).detach().numpy()
#     output_streaming = streaming_inference(model_stream, X, embedding=EMB, chunk_size=CHUNK_SIZE, pad_length=PAD_FRONT).detach().numpy()

# print(output_streaming.shape, output_full.shape)
# assert output_full.shape == output_streaming.shape

# os.makedirs('pretrained/test_data/streaming_test', exist_ok=True)

# np.save('pretrained/test_data/streaming_test/e2e_input_X.npy', arr=X.detach().numpy())
# np.save('pretrained/test_data/streaming_test/e2e_output_streaming.npy', arr=output_streaming)
# np.save('pretrained/test_data/streaming_test/e2e_output_full.npy', arr=output_full)

# # print(list(output_streaming[0, 0, :10]))
# print("Test successful:", np.allclose(output_streaming, output_full, atol=1e-3))
# print("Max diff:", np.max(np.abs(output_streaming - output_full)))
