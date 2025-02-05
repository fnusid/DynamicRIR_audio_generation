import torch.nn.functional as F
import os, glob
from pathlib import Path
import random
import scipy.ndimage
from torch.utils.data import Dataset
import pandas as pd
import torch
import torchaudio
import tqdm
from multiprocessing import Pool
# from src.datasets.augmentations.audio_augmentations import AudioAugmentations
import numpy as np
import subprocess
from gpuRIRsimulateRIR import PRASimulator
# from scipy.signal import fftconvolve
import pdb
import soundfile as sf


def __getaudio(path=Path('/mmfs1/gscratch/intelligentsystems/common_datasets/spatial_audio/train')):
    path_files = sorted(list(path.glob('**/*.wav')))
    audios = []
    for audio_path in tqdm.tqdm(path_files):

        audio, sr = torchaudio.load(audio_path)
        audios.append(audio)
    audios_tensor = torch.stack(audios)
    mean_vec = torch.mean(audios_tensor, dim=0)
    std_vec = torch.std(audios_tensor, dim=0)
    np.savetxt("/mmfs1/gscratch/intelligentsystems/sidharth/codebase/ml_pipeline/spatial_audio_expt/src/datasets/std.txt", std_vec.numpy())
    np.savetxt("/mmfs1/gscratch/intelligentsystems/sidharth/codebase/ml_pipeline/spatial_audio_expt/src/datasets/mean.txt", mean_vec.numpy())





if __name__=='__main__':
    __getaudio(Path('/mmfs1/gscratch/intelligentsystems/common_datasets/spatial_audio/train'))