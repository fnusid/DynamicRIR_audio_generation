"""
Torch dataset object for synthetically rendered spatial data.
"""

import os, glob
from pathlib import Path
import random
from torch.utils.data import Dataset
import pandas as pd
import torch
import torchaudio
from src.datasets.augmentations.audio_augmentations import AudioAugmentations
import numpy as np


class LookOnceToHearDataset(Dataset):
    def __init__(self, dataset_dir, sr, split, augmentations = [], max_samples=None) -> None:
        super().__init__()
        assert split in ['train', 'val'], \
            "`split` must be one of ['train', 'val']"

        self.dataset_dir = dataset_dir
        self.split = split
        self.sr = sr

        # Data augmentation
        self.perturbations = AudioAugmentations(augmentations)

        self.samples = sorted(list(Path(self.dataset_dir).glob('[0-9]*')))
        self.max_samples = max_samples
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]

        mixture_path = os.path.join(sample_dir, 'mixture.wav')
        mixture, sample_rate = torchaudio.load(mixture_path)
        assert sample_rate == self.sr
        
        gt_path = os.path.join(sample_dir, 'gt.wav')
        gt, sample_rate = torchaudio.load(gt_path)
        assert sample_rate == self.sr

        enrollment_path = os.path.join(sample_dir, 'enrollment.wav')
        enrollment, sample_rate = torchaudio.load(enrollment_path)
        assert sample_rate == self.sr
        
        embedding = np.load(os.path.join(sample_dir, 'embedding.npy')) # Should be label_vector.npy but I had a bug
        embedding = torch.from_numpy(embedding)

        # Apply perturbations to entire audio
        if self.split == 'train':
            mixture, gt = self.perturbations.apply_random_augmentations(mixture, gt)

        # Generate mixture and gt audio        
        peak = torch.abs(mixture).max()
        if peak > 1:
            mixture /= peak
            gt /= peak
        
        inputs = {
            'mixture': mixture,
            'enrollment': enrollment,
            "sample_dir":sample_dir.absolute().as_posix()
        }

        targets = {
            "target": gt,
            'embedding_gt': embedding,
            "num_target_speakers":1
        }

        return inputs, targets
