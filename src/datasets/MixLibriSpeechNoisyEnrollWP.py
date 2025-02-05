"""
Torch dataset object for synthetically rendered spatial data.
"""

import os, glob
from pathlib import Path
import random
import logging
import warnings
import pdb
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
    def __init__(self, fg_dir, bg_dir, embed_dir, jams_dir, hrtf_list, split='val',
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
        # breakpoint()
        self.samples = sorted(list(Path(self.jams_dir).glob('[0-9]*')))[:max_samples]
        self.max_samples = max_samples
        # breakpoint()
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
        breakpoint()
        # Load Audio
        jamsfile = os.path.join(sample_dir, 'mixture.jams')
        _, jams, ann_list, event_audio_list = scaper.generate_from_jams(
            jamsfile, fg_path=self.fg_dir, bg_path=self.bg_dir)

        for x in event_audio_list:
            assert x.shape[1] == 1, "Number of channels is 1"

        # MIXTURE ================================================================================================
        # Squeeze to mono
        event_audio_list = [x.squeeze(1) for x in event_audio_list]

        # Simulate multi-channel audio
        multi_ch_seed = idx
        if self.split == 'train':
            multi_ch_seed = random.randrange(1, 100000)
        multi_ch_events, multi_ch_noise = self.multi_ch_simulator.simulate(
            event_audio_list[1:], event_audio_list[0], multi_ch_seed)

        # To torch
        multi_ch_events = [torch.from_numpy(x).float() for x in multi_ch_events]
        multi_ch_noise = torch.from_numpy(multi_ch_noise).float()
        
        # Select target index
        if self.split == 'train':
            tgt_idx_in_mixture_events = random.randrange(len(multi_ch_events))
        else:
            _rng = random.Random(idx)
            tgt_idx_in_mixture_events = _rng.randrange(len(multi_ch_events))

        # Target id
        # breakpoint()
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
        
        # Save speaker info
        other_spk_info = []
        for sf in source_files[1:]: # skip background
            if sf == 'None':
                other_spk_info.append(('None', 'None'))
                continue
            spk_id = os.path.basename(sf).split('-')[0]
            if spk_id != tgt_id:
                other_spk_info.append((spk_id, self.speaker_info[spk_id]))
        speaker_info = [(tgt_id, self.speaker_info[tgt_id])] + other_spk_info
        # DEBUGGING PURPOSES  ================================================================================================
        
        # Get mixture and gt
        mixture = sum(multi_ch_events) + multi_ch_noise
        gt = multi_ch_events[tgt_idx_in_mixture_events].clone()

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
        
        # embedding_neg = []
        # for i, sf in enumerate(enroll_source_files[1:]):
        #     if sf == 'None':
        #         embedding_neg.append(torch.zeros_like(embedding_neg[-1]))
        #         continue
        #     filename = os.path.basename(sf)
        #     spk_id = filename.split('-')[0]
        #     if spk_id != tgt_id:
        #         embedding_neg.append(self._get_dvector_embedding(spk_id, filename))

        if self.skip_enrollment_simulation:
            # Use monaural signal as "enrollment"
            mono = torch.from_numpy(enroll_event_audio_list[1:][enroll_target_idx])
            enroll = torch.stack([mono, mono], dim=0)
            assert enroll.shape == gt.shape
            
            enroll_clean = enroll
        else:
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
        mixture = self.resampler(mixture)
        target = self.resampler(gt)
        enroll = self.resampler(enroll)
        
        # Apply perturbations to entire audio
        if self.split == 'train':
            mixture, target = self.perturbations.apply_random_augmentations(mixture, target)
        
        # Normalize enrollment audio
        peak = torch.abs(enroll).max()
        if peak > 1:
            enroll /= peak

        # Normalize mixture audio
        peak = torch.abs(mixture).max()
        if peak > 1:
            mixture /= peak
            gt /= peak

        inputs = {
            'mixture': mixture,
            'enrollment': enroll,
            # 'enrollments_clean': enroll_clean.unsqueeze(0),
            # 'enrollments_clean_path': [enroll_clean_path],
            # 'enrollments_id': torch.tensor([int(tgt_id)]),
            # 'enrollments_source_files': enroll_source_files,
            'source_files': source_files,
            'speaker_info': speaker_info,
            "sample_dir":sample_dir.absolute().as_posix()
        }

        targets = {
            'target': target,
            'embedding_gt': embedding_gt.unsqueeze(0),
            'tgt_spk_idx': tgt_id,
            "num_target_speakers": 1
        }

        return inputs, targets
if __name__ == '__main__':
    fg_dir = '/scr/MixLibriSpeech/librispeech_scaper_fmt/dev-clean'
    bg_dir = '/scr/MixLibriSpeech/wham_noise'
    jams_dir = '/scr/MixLibriSpeech/jams/dev-clean'
    embed_dir = '/scr/MixLibriSpeech/librispeech_dvector_embeddings/dev-clean'
    hrtf_list = [
            "/scr/MixLibriSpeech/CIPIC/train_hrtf.txt",
            "/scr/RRBRIR/train_hrtf.txt",
            "/scr/ASH-Listening-Set-8.0/BRIRs",
            "/scr/CATT_RIRs/Binaural/16k"
        ]
    hrtf_type =  'MultiCh'
    num_enroll =  1
    sr =  16000
    max_samples=2000
    skip_enrollment_simulation = False
    use_motion = False
    split= 'val'
    params = {
    'fg_dir': fg_dir,
    'bg_dir': bg_dir,
    'jams_dir': jams_dir,
    'embed_dir': embed_dir,
    'hrtf_list': hrtf_list,
    'hrtf_type': hrtf_type,
    'num_enroll': num_enroll,
    'sr': sr,
    'max_samples': max_samples,
    'skip_enrollment_simulation': skip_enrollment_simulation,
    'use_motion': use_motion,
    'split': split
    }
    dataset = MixLibriSpeechNoisyEnroll(**params)
    dataset.__getitem__(1)

