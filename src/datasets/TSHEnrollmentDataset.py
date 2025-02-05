"""
Torch dataset object for synthetically rendered spatial data.
"""

import os, glob
from pathlib import Path
import random
import logging
import warnings

import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio.transforms as AT
import scaper

from src.datasets.augmentations.audio_augmentations import AudioAugmentations

from src.datasets.motion_simulator import CIPICMotionSimulator2
from src.datasets.multi_ch_simulator import (
    CIPICSimulator, APLSimulator, RRBRIRSimulator, ASHSimulator, CATTRIRSimulator, MultiChSimulator)

# Ignore scaper normalization warnings
warnings.filterwarnings(
    "ignore", message="Soundscape audio is clipping!")
warnings.filterwarnings(
    "ignore", message="Peak normalization applied")
warnings.filterwarnings(
    "ignore", message="Scale factor for peak normalization is extreme")

class MixLibriSpeechNoisyEnroll(Dataset):
    def __init__(self, fg_dir, bg_dir, embed_dir, jams_dir, hrtf_list, split,
                 sr=None, resample_rate=None, num_enroll=10, hrtf_type="CIPIC",
                 skip_enrollment_simulation=False,
                 use_motion=False, motion_use_piecewise_arcs=False,
                 augmentations = [], max_samples=20000) -> None:
        super().__init__()
        assert split in ['train', 'val'], \
            "`split` must be one of ['train', 'val']"

        self.fg_dir = fg_dir
        self.bg_dir = bg_dir
        self.hrtf_list = hrtf_list
        self.jams_dir = jams_dir
        self.embed_dir = embed_dir
        self.split = split
        self.sr = sr
        self.resample_rate = resample_rate
        self.num_enroll = num_enroll
        self.skip_enrollment_simulation = skip_enrollment_simulation

        # Data augmentation
        self.perturbations = AudioAugmentations(augmentations)

        logging.info(f"Loading dataset: {split} {sr=} {resample_rate=} ...")
        logging.info(f"  - Foreground directory: {fg_dir}")
        logging.info(f"  - Background directory: {bg_dir}")
        logging.info(f"  - Embedding directory: {embed_dir}")
        logging.info(f"  - JAMS directory: {jams_dir}")
        logging.info(f"  - HRTF directory: {hrtf_list}")

        self.samples = sorted(list(Path(self.jams_dir).glob('[0-9]*')))[:max_samples]
        self.max_samples = max_samples

        if resample_rate is not None:
            self.resampler = AT.Resample(sr, resample_rate)
            self.sr = resample_rate
        else:
            self.resampler = lambda a: a
            self.sr = sr

        # Get speaker info
        speaker_txt = os.path.join(self.fg_dir, '..', '..', 'LibriSpeech', 'SPEAKERS.TXT')
        self.speaker_info = self._get_speaker_info(speaker_txt)

        # HRTF simulator with motion
        if hrtf_type == 'CIPIC':
            self.multi_ch_simulator = CIPICSimulator(self.hrtf_list, sr)
        elif hrtf_type == 'APL':
            self.multi_ch_simulator = APLSimulator(self.hrtf_list, sr)
        elif hrtf_type == 'ASH':
            self.multi_ch_simulator = ASHSimulator(self.hrtf_list, sr, dset=split)
        elif hrtf_type == 'CATTRIR':
            self.multi_ch_simulator = CATTRIRSimulator(self.hrtf_list, sr, dset=split)
        elif hrtf_type == 'RRBRIR':
            self.multi_ch_simulator = RRBRIRSimulator(self.hrtf_list, sr)
        elif hrtf_type == 'MultiCh':
            if use_motion:
                cipic_simulator_type = \
                    lambda sofa, sr: CIPICMotionSimulator2(sofa, sr, use_piecewise_arcs=motion_use_piecewise_arcs)
            else:
                cipic_simulator_type = CIPICSimulator
            
            self.multi_ch_simulator = MultiChSimulator(self.hrtf_list, sr, cipic_simulator_type, dset=split)
        elif hrtf_type == 'CIPIC_MOTION':
            self.multi_ch_simulator = CIPICMotionSimulator2(self.hrtf_list, sr)
        else:
            raise NotImplementedError

        # Create speaker to samples mapping
        self.speaker_map = {}
        for i, sample_dir in enumerate(self.samples):
            anns = pd.read_csv(os.path.join(sample_dir, 'mixture.txt'), sep='\t', header=None)
            speaker_ids = anns[2]
            for spk_id in speaker_ids:
                if spk_id not in self.speaker_map:
                    self.speaker_map[spk_id] = [i]
                else:
                    self.speaker_map[spk_id].append(i)

        assert num_enroll == 1, "Only 1 enrollment is supported"

        # Speaker ids
        self.speaker_ids = os.listdir(self.fg_dir)
        self.speaker_ids = [int(x) for x in self.speaker_ids]

    def __len__(self):
        return len(self.samples)

    def _get_speaker_info(self, speaker_txt):
        # Read librispeech speaker info to a dataframe
        with open(speaker_txt) as f:
            lines = f.readlines()
        lines = lines[11:]

        # Creae dataframe from lines
        df = pd.DataFrame([l.strip().split('|') for l in lines])

        # Remove extra whitespaces throughout the dataframe
        df = df.iloc[1:]
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

        speaker_info = {}
        for i, row in df.iterrows():
            speaker_info[row[df.columns[0]]] = row[df.columns[1]]

        return speaker_info

    def _get_dvector_embedding(self, spk_id, filename):
        embed_map = torch.load(
            os.path.join(self.embed_dir, f'{spk_id}.pt'), map_location='cpu')
        return torch.from_numpy(embed_map[filename])

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]
        multi_ch_seed = idx

        # Load Audio
        jamsfile = os.path.join(sample_dir, 'mixture.jams')
        _, jams, ann_list, event_audio_list = scaper.generate_from_jams(
            jamsfile, fg_path=self.fg_dir, bg_path=self.bg_dir)
        
        # Select target index
        if self.split == 'train':
            tgt_idx_in_mixture_events = random.randrange(len(ann_list))
        else:
            _rng = random.Random(idx)
            tgt_idx_in_mixture_events = _rng.randrange(len(ann_list))
        
        # Target id
        tgt_id = ann_list[tgt_idx_in_mixture_events][-1]

        # Record source files
        # DEBUGGING PURPOSES  ================================================================================================
        # Source file list and speaker info
        source_files = []
        for obv in jams.annotations[0].data:
            source_files.append(obv.value['source_file'])

        # Pad source files to length 3 for collate_fn
        if len(source_files) == 3:
            source_files.append('None')
        
        # ENROLLMENT ================================================================================================
        # Select and enrollment sample using the speaker map
        if self.split == 'train':
            enroll_id = random.choice(self.speaker_map[int(tgt_id)])
        else:
            _rng = random.Random(idx)
            enroll_id = _rng.choice(self.speaker_map[int(tgt_id)])
        enroll_dir = self.samples[enroll_id]
        
        # Load enrollment audio
        enroll_jams = os.path.join(enroll_dir, 'mixture.jams')
        enroll_spks = pd.read_csv(
            os.path.join(enroll_dir, 'mixture.txt'), sep='\t', header=None)[2].tolist()
        _, enroll_jams, enroll_anns, enroll_event_audio_list = scaper.generate_from_jams(
            enroll_jams, fg_path=self.fg_dir, bg_path=self.bg_dir)
        enroll_source_files = []
        for obv in enroll_jams.annotations[0].data:
            enroll_source_files.append(obv.value['source_file'])
        if len(enroll_source_files) == 3:
            enroll_source_files.append('None')
        
        # Squeeze to mono
        enroll_event_audio_list = [x.squeeze(1) for x in enroll_event_audio_list]

        # Clean enrollment
        enroll_target_idx = enroll_spks.index(int(tgt_id))
        enroll_clean_path = enroll_jams.annotations[0].data[enroll_target_idx + 1].value['source_file']

        # Ground truth embeddings and negative embeddings
        embedding_gt = self._get_dvector_embedding(tgt_id, os.path.basename(enroll_clean_path))

        # Simulate multi-channel audio
        enroll_multi_ch_events, enroll_multi_ch_noise = self.multi_ch_simulator.simulate(
            enroll_event_audio_list[1:], enroll_event_audio_list[0], multi_ch_seed,
            face_to_face_idx=enroll_target_idx)
        enroll_clean = enroll_multi_ch_events[enroll_target_idx]
        enroll_clean = torch.from_numpy(enroll_clean).float()
        enroll_event_audio_list = [
            torch.from_numpy(x).float() for x in enroll_multi_ch_events
        ]
        enroll_event_audio_list += [torch.from_numpy(enroll_multi_ch_noise).float()]
        enroll = sum(enroll_event_audio_list)

        # Resample
        embedding_gt = self.resampler(embedding_gt)
        enroll = self.resampler(enroll)
        
        # Normalize enrollment audio
        peak = torch.abs(enroll).max()
        if peak > 1:
            enroll /= peak

        inputs = {
            'mixture': enroll
        }

        targets = {
            'embedding': embedding_gt.unsqueeze(0),
            'enrollment_clean': enroll_clean,
            'tgt_spk_idx': tgt_id,
            "num_target_speakers": 1
        }

        return inputs, targets
