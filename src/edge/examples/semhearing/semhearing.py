import sounddevice as sd
import queue
import numpy as np
import torch
import threading
import onnxruntime as ort
from src.edge.edge_utils import flatten_state_buffers
from scipy.io.wavfile import write as wavwrite
import time
from src.datasets.augmentations.WhitePinkBrownAugmentation import powerlaw_psd_gaussian
import soundfile as sf
import os
from datetime import datetime

labels = [
            "cat", "cock_a_doodle_doo", "cricket", "dog", "door_knock",
        ]

def show_labels():
    for i, label in enumerate(labels):
        print(f"[{i}] {label}")
    print()

def write_audio_file(file_path, data, sr):
    """
    Writes audio file to system memory.
    @param file_path: Path of the file to write to
    @param data: Audio signal to write (n_channels x n_samples)
    @param sr: Sampling rate
    """
    wavwrite(file_path, sr, data.T)

# STREAM PARAMS
fname = 'recording.wav'

sr = 44100 # Sampling rate of input audio
tgt_sr = 44100 # Sampling rate of processing audio

downsample_rate = sr // tgt_sr

# Embedding model will run on pytorch
from src.utils import load_net
# _model, model_params = load_net('runs/semhearing_tfgridnet_large_short_with_asserts/config.json', return_params=True)
#_model, model_params = load_net('runs/semhearing_tfgridnet_smaller_short_samples/config.json', return_params=True)
_model, model_params = load_net('runs/semhearing_LONG_SAMPLES/config.json', return_params=True)
_model = _model.model

state_buffers = _model.init_buffers(1, 'cpu')
tse_path = 'pretrained/ONNX/model.onnx'

model_params = model_params['pl_module_args']['model_params']
CHUNK_SIZE = model_params['stft_chunk_size']
PAD_SIZE = model_params['stft_pad_size']

BLOCKSIZE = CHUNK_SIZE
print(BLOCKSIZE)
target_acquired = False
running = 0
overrun = 0

in_queue = queue.Queue()
ZEROS = np.zeros((BLOCKSIZE, 2), dtype=np.float32)
OUT_BUF = None

pink_noise = powerlaw_psd_gaussian(1, (2, 5 * sr), random_state = 0) * 0.001
brown_noise = powerlaw_psd_gaussian(2, (2, 5 * sr), random_state = 0) * 0.005
colored_noise = powerlaw_psd_gaussian(3, (2, 5 * sr), random_state = 0) * 0.005
noise = pink_noise + brown_noise + colored_noise
noise = noise * 5
noise = [noise[:, i:i+CHUNK_SIZE] for i in range(0, noise.shape[-1] - CHUNK_SIZE, 256)]

class ModelWrapper():
    def __init__(self, tse_path, state_buffers):
        self.buffer_names, self.buffers = flatten_state_buffers(state_buffers)
        
        self.current_inputs = dict()
        for buf_name, buf in zip(self.buffer_names, self.buffers):
            self.current_inputs[buf_name] = buf.detach().numpy()
            print(buf_name, buf.shape)

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2

        self.tse = ort.InferenceSession(tse_path, providers=['CPUExecutionProvider'], sess_options=sess_options)

        self.embedding = None

        # # Cold start
        # self.tse.run(None, self.current_inputs)

    def set_target(self, label: str):
        global labels, target_acquired
        
        vector = torch.zeros(len(labels))
        idx = labels.index(label)
        assert vector[idx] == 0, "Repeated labels"
        vector[idx] = 1
        
        self.embedding = vector.unsqueeze(0).unsqueeze(1)
        
        self.current_inputs['embedding'] = self.embedding.numpy()
        for buf_name, buf in zip(self.buffer_names, self.buffers):
            self.current_inputs[buf_name] *= 0#buf.detach().numpy()
        print("EMBEDDING FINISHED")

        target_acquired = True

    def infer(self, x: np.ndarray) -> np.ndarray:
        if self.embedding is None:
            print("EMBEDDING IS NONE")
            return x[..., :-PAD_SIZE]
        
        self.current_inputs['mixture'] = x
        
        t1 = time.time()
        outputs = self.tse.run(None, self.current_inputs)
        t2 = time.time()
        #if t2 - t1 > 8e-3:
         #   print('TOO LONG', t2 - t1)
        
        for i in range(1, len(self.buffer_names)+1):
            # print(self.buffer_names[i-1])
            #print(self.buffer_names[i-1], outputs[i].shape)
            self.current_inputs[self.buffer_names[i-1]] = outputs[i]

        return outputs[0]

def stream_callback(indata, outdata, frames, time, status):
    global OUT_BUF
    if OUT_BUF is not None:
        outdata[:] = OUT_BUF.T
        OUT_BUF = None
    else:
        outdata[:] = ZEROS
    
    if in_queue.qsize() > 0:
        with in_queue.mutex:
            in_queue.queue.clear()
    in_queue.put(indata.T.copy())

# Initialize stream
stream = sd.Stream(samplerate=sr,
                   latency='low',
                    device=(3, 2),
                   channels=(2, 2),
                   blocksize=BLOCKSIZE,
                   callback=stream_callback,
                   dtype=np.float32)


model = ModelWrapper(tse_path, state_buffers)

def process_enrollment_in_background(enrollment):
    # enrollment = resample(enrollment, enrollment.shape[-1] // downsample_rate, axis=-1).astype(np.float32)
    enrollment = enrollment.astype(np.float32)
    threading.Thread(target=model.enroll, args=(enrollment,)).start()

def stop():
    global running
    running = False
    stream.stop()
    stream.close()

def cmd_input(message):
    cmdlist = input(message).strip().split(' ')
    cmdlist = [x for x in cmdlist if x != '']
    
    if len(cmdlist) == 0:
        return cmd_input(message)
    
    if cmdlist[0] == 'cancel':
        print("Cancelled")
        return None
    
    return cmdlist

started = threading.Event()

def respond_to_inputs():
    global labels, target_acquired
    started.set()
    
    while True:
        cmdlist = cmd_input("Enter command [label|pause|unpause|quit]:")

        cmd = cmdlist[0]
        if cmd == 'quit':
            stop()
            break
        elif cmd == 'label':
            show_labels()
            cmdlist = cmd_input("Label [id]:")
            label_id = int(cmdlist[0])
            model.set_target(labels[label_id])
        elif cmd == 'pause':
            target_acquired = False
        elif cmd == 'unpause':
            target_acquired = True
        else:
            print(f"Command {cmd} not recognized")


from flask import Flask
from flask import request
from flask import render_template

import json

app = Flask(__name__, static_url_path='', static_folder='res')

@app.route('/', methods = ['GET', 'POST'])
def user():
    global labels, target_acquired
    if request.method == 'POST':
        data = json.loads(request.data.decode()) # a multidict containing POST data
        class_id = data['class_id'][:-4]
        print("Received selection:", class_id)

        if class_id == 'stop':
            stop()
        elif class_id == 'pause':
            target_acquired = False
        else:
            if class_id == 'rooster':
                class_id = 'cock_a_doodle_doo'
            elif class_id == 'knock':
                class_id = 'door_knock'
            model.set_target(class_id)
    
    return render_template('index.html')
stopped = False
def run_semantic_hearing():
    global OUT_BUF, running
    # p = pyaudio.PyAudio()
    # incoming_stream_file = p.open(format=p.get_format_from_width(2),
    #                                 channels=2,
    #                                 rate=16000,
    #                                 output=True)
    # p = pyaudio.PyAudio()
    # playback_stream_file = p.open(format=p.get_format_from_width(2),
    #                                 channels=2,
    #                                 rate=16000,
    #                                 output=True)
    
    formatted_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M")
    write_dir = f'saved_audio/run_{formatted_datetime}'
    os.makedirs(write_dir, exist_ok=True)
    sf_incoming = sf.SoundFile(os.path.join(write_dir, 'incoming.wav'), mode='w', samplerate=44100, channels=2, subtype='PCM_16')
    sf_playback = sf.SoundFile(os.path.join(write_dir, 'output.wav'), mode='w', samplerate=44100, channels=2, subtype='PCM_16')

    print("SEMANTIC HEARING IS RUNNING")

    running = True
    
    odata = []
    
    T = CHUNK_SIZE
    
    current_frame = np.zeros((1, 2, T + PAD_SIZE), dtype=np.float32)
    data = []
    odata = []

    stream.start()
    noise_idx = 0
    while running:
        try:
            input_data = in_queue.get(timeout=0.015) * 2
            peak = np.abs(input_data).max()
            if peak > 1:
                input_data /= peak

            if target_acquired:
                # Resample to 16kHz
                # input_data_resampled = resample(input_data, CHUNK_SIZE, axis=-1)
                input_data_resampled = input_data
                current_frame = np.roll(current_frame, shift=-T, axis=-1)
                current_frame[0, :, -T:] = input_data_resampled

                y = model.infer(current_frame)[0]
                #y = input_data_resampled
            else:                
                y = input_data
            
            sf_incoming.write(input_data.T)
            sf_playback.write(y.T)

            if target_acquired:
                y = y * 5 + noise[noise_idx]
                noise_idx = (noise_idx + 1) % len(noise)

            
            OUT_BUF = y
            # incoming_stream_file.write(input_data)
            # playback_stream_file.write(y)
            # data.append(input_data)
            # odata.append(y)
        except queue.Empty:
            pass
    
    print("STOPPED")
    
    sf_incoming.close()
    sf_playback.close()

# while True:
threading.Thread(target=run_semantic_hearing).start()
app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False)
stream.stop()
stopped = True
    # started.wait()
    
    # data = np.concatenate(data, axis=1)
    # odata = np.concatenate(odata, axis=1)
    # print(data.shape)
        
    # write_audio_file('saved_input.wav', data, sr)
    # write_audio_file('saved_output.wav', odata, sr)


    
