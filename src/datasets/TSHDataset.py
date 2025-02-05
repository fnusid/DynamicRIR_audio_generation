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
from src.datasets.multi_ch_simulator import (CIPICSimulator, MultiChSimulator)
import numpy as np
import soundfile as sf
import copy


# Ignore scaper normalization warnings
warnings.filterwarnings(
    "ignore", message="Soundscape audio is clipping!")
warnings.filterwarnings(
    "ignore", message="Peak normalization applied")
warnings.filterwarnings(
    "ignore", message="Scale factor for peak normalization is extreme")


def get_snr(target, mixture, EPS=1e-9):
    """
    Computes the average SNR across both channels of a BINAURAL signal
    """
    snr = lambda s, n: 10 * (torch.log10 (s.dot(s) + EPS) - torch.log10 (n.dot(n) + EPS))
    
    snr_left = snr(target[0], mixture[0] - target[0])
    snr_right = snr(target[1], mixture[1] - target[1])
    
    return (snr_left + snr_right).item() / 2

def scale_noise_to_snr(target_speech: torch.Tensor, noise: torch.Tensor, target_snr: float):
    """
    Rescales a BINAURAL noise signal to achieve an average SNR (across both channels) equal to target snr.
    Let k be the noise scaling factor
    SNR_tgt = (SNR_left_scaled + SNR_right_scaled) / 2 = 0.5 * (10 log(S_L^T S_L/S_N^T S_N) - 20 log(k) + 10 log(S_R^T S_R / N_R^T N_R) - 20 log(k))
            = 0.5 * (SNR_left_unscaled + SNR_right_unscaled - 40 log(k)) = avg_snr_initial - 20 log (k)
    """
    
    current_snr = get_snr(target_speech, noise + target_speech)

    pwr = (current_snr - target_snr) / 20
    k = 10 ** pwr

    return k * noise


# TODO:
# Add data augmentation
# Remove gt peak normalization

class MixLibriSpeechNoisyEnroll(Dataset):
    def __init__(self, speech_dir, noise_dir, hrtf_list, embed_dir, split,
                 duration = 5, sr=16000, num_enroll=10, hrtf_type="CIPIC",
                 skip_enrollment_simulation=False,
                 motion_use_piecewise_arcs=False,
                 mix_snr_min = -10, mix_snr_max = 5,
                 enr_snr_min = -10, enr_snr_max = 5,
                 augmentations = [], samples_per_epoch=20000, use_motion = False,
                 num_speakers_min=1, num_speakers_max=4, num_noise_srcs=1) -> None:
        super().__init__()
        assert split in ['train', 'val'], \
            "`split` must be one of ['train', 'val']"

        self.speech_dir = speech_dir
        self.noise_dir = noise_dir
        self.hrtf_list = hrtf_list
        self.embed_dir = embed_dir
        self.split = split
        self.sr = sr
        self.num_enroll = num_enroll
        self.skip_enrollment_simulation = skip_enrollment_simulation
        self.duration = duration
        self.mix_snr_range = [mix_snr_min, mix_snr_max]
        self.enr_snr_range = [enr_snr_min, enr_snr_max]
        self.num_speakers_min = num_speakers_min
        self.num_speakers_max = num_speakers_max
        self.samples_per_epoch = samples_per_epoch
        self.num_noise_srcs = num_noise_srcs

        # Data augmentation
        self.perturbations = AudioAugmentations(augmentations)

        logging.info(f"  - Speech directory: {speech_dir}")
        logging.info(f"  - Noise directory: {noise_dir}")
        logging.info(f"  - Embedding directory: {embed_dir}")
        logging.info(f"  - HRTF directory: {hrtf_list}")

        # HRTF simulator with motion
        if hrtf_type == 'CIPIC':
            self.multi_ch_simulator = CIPICSimulator(self.hrtf_list, sr)
        elif hrtf_type == 'MultiCh':
            if use_motion:
                cipic_simulator_type = \
                    lambda sofa, sr: CIPICMotionSimulator2(sofa, sr, use_piecewise_arcs=motion_use_piecewise_arcs)
            else:
                cipic_simulator_type = CIPICSimulator
            
            self.multi_ch_simulator = MultiChSimulator(self.hrtf_list, sr, cipic_simulator_type, dset=split)
        else:
            raise NotImplementedError

        assert num_enroll == 1, "Only 1 enrollment is supported"

        # Speaker ids
        self.speaker_ids = os.listdir(self.speech_dir)
        self.speaker_ids = [int(x) for x in self.speaker_ids]

    def __len__(self):
        return self.samples_per_epoch

    def _get_dvector_embedding(self, spk_id, filename):
        embed_map = torch.load(
            os.path.join(self.embed_dir, f'{spk_id}.pt'), map_location='cpu')
        return torch.from_numpy(embed_map[filename])

    def read_audio_as_mono(self, sf: sf.SoundFile, num_frames: int):
        # Workaround for loading flac files
        if sf.name.endswith('.flac'):
            audio = sf.read(frames=num_frames, dtype='int32')
            audio = (audio / (2 ** (31) - 1)).astype(np.float32)
        else:
            audio = sf.read(frames=num_frames, dtype='float32')
        
        # If multichannel audio take a single channel
        if len(audio.shape) > 1:
            audio = audio[:, 0]

        return audio
    
    def sample_snippet(self, audio_file, rng):
        try:
            with sf.SoundFile(audio_file) as f:
                file_sr = f.samplerate
                
                assert file_sr == self.sr, "File sampling rate doesn't match expected sampling rate!"
                
                tgt_samples = int(round(file_sr * self.duration))
                num_frames = f.frames
        
                if tgt_samples > num_frames:
                    # Read entire audio
                    audio = self.read_audio_as_mono(f, num_frames=num_frames)
        
                    # Pad zeros to get to target duration
                    remain = tgt_samples - num_frames
                    pad_front = rng.randint(0, remain)
                    audio = np.pad(audio, (pad_front, remain - pad_front))
        
                else:
                    # Randomly choose start of snippet
                    start_frame = np.random.randint(0, num_frames - tgt_samples + 1)
        
                    # Move to start of snippet and read
                    f.seek(start_frame)
                    audio = self.read_audio_as_mono(f, num_frames=tgt_samples)
        
                assert audio.shape[-1] == tgt_samples, f"Number of samples in audio incorrect.\
                                                         Expceted {audio.shape[-1]} found {tgt_samples}."
        except:
            print("ERROR AT FILE: ", audio_file, start_frame)
            return self.sample_snippet(audio_file, rng)
        return audio

    def get_random_noise_snippet(self, rng: np.random.RandomState):
        noise_audio_list = glob.glob(os.path.join(self.noise_dir, '*.wav'))
        noise_path = rng.choice(noise_audio_list, 1)[0]
        
        return self.sample_snippet(noise_path, rng)

    def get_random_speaker_snippet(self, speaker_id: int, rng: np.random.RandomState):
        speaker_path = os.path.join(self.speech_dir, str(speaker_id))
        speaker_audio_list = glob.glob(os.path.join(speaker_path, '*.flac'))
        
        speaker_audio_path = rng.choice(speaker_audio_list, 1)[0]
        
        return speaker_audio_path, self.sample_snippet(speaker_audio_path, rng)

    def _get_random_tgt_enrollment(self, spk_mix_path, spk_id, rng: np.random.RandomState):
        embed_map = torch.load(os.path.join(self.embed_dir, f'{spk_id}.pt'), map_location='cpu')
        fnames = list(embed_map.keys())
        if spk_mix_path is not None:
            fname_in_mixture = os.path.basename(spk_mix_path)
            assert fname_in_mixture in fnames
            fnames.remove(fname_in_mixture)
        
        # Choose random enrollment
        enrollment_fname = rng.choice(fnames, size=1)[0]

        # Get embedding
        embedding = torch.from_numpy(embed_map[enrollment_fname])

        # Get corresponding enrollment
        speaker_path = os.path.join(self.speech_dir, str(spk_id), enrollment_fname)
        assert os.path.exists(speaker_path), f"Enrollment path {speaker_path} not found!"
        
        enrollment = self.sample_snippet(speaker_path, rng)

        # if True:
        #     from resemblyzer import VoiceEncoder, preprocess_wav
        #     import numpy as np
        #     wav = preprocess_wav(speaker_path)
        #     encoder = VoiceEncoder()
        #     embed = encoder.embed_utterance(wav)
        #     embed = torch.from_numpy(embed)
        #     print(embed.shape)
        #     print(embedding.shape)
        #     print("Max diff:", torch.abs(embed - embedding).max())
        #     assert torch.allclose(embed, embedding), "Embeddings are not the same!"

        assert enrollment is not None, f"Could not find enrollment audio for file {enrollment_fname}"

        return embedding, enrollment

    def create_scene(self, rng: np.random.RandomState,
                     enrollment = False, enrollment_id = None, spk_mix_path=None):
        speakers = copy.deepcopy(self.speaker_ids)
        speaker_ids = []
        
        if enrollment:
            assert enrollment_id is not None, "For enrollment, an target id must be given"
            speakers.remove(enrollment_id) # Do not randomly choose this speaker
            speaker_ids.append(enrollment_id) # Force choose
        else:
            assert enrollment_id is None, "For TSH, no target id should be given"

        # Sample number of speakers
        # Get 1 less speaker
        n_speakers = rng.randint(self.num_speakers_min, self.num_speakers_max + 1)

        # Randomly speaker ids
        speaker_ids = speaker_ids + rng.choice(speakers, size=n_speakers - enrollment, replace=False).tolist()

        # Sanity check
        assert len(speaker_ids) == n_speakers
        assert len(speakers) == len(self.speaker_ids) - enrollment

        # Load random audio for each speaker
        speaker_audio = []

        embedding = None
        for i, speaker_id in enumerate(speaker_ids):
            
            if enrollment and (i == 0):
                assert embedding is None
                embedding, spk_audio = self._get_random_tgt_enrollment(spk_mix_path, speaker_id, rng)
            else:
                spk_path, spk_audio = self.get_random_speaker_snippet(speaker_id, rng)
                
                if i == 0:
                    # Sanity check
                    assert spk_mix_path is None
                    
                    spk_mix_path = spk_path

            speaker_audio.append(spk_audio)

        # Sanity check
        assert spk_mix_path is not None
        
        # Load random noise file
        noise_audio = self.get_random_noise_snippet(rng)

        # Simulate
        seed = rng.randint(1, 1000000)
        if enrollment:
            # For enrollment, target should be facing user
            multi_ch_events, multi_ch_noise = self.multi_ch_simulator.simulate(speaker_audio, noise_audio, seed, face_to_face_idx=0)
        else:
            multi_ch_events, multi_ch_noise = self.multi_ch_simulator.simulate(speaker_audio, noise_audio, seed)

        # To torch
        multi_ch_events = [torch.from_numpy(x).float() for x in multi_ch_events]
        multi_ch_noise = torch.from_numpy(multi_ch_noise).float()
        
        # Set the first speaker to be the target
        tgt_id = speaker_ids[0]
        
        # Get mixture and gt 
        # Randomly scale each audio event
        for i in range(1, len(multi_ch_events)):
            scale = rng.random()
            multi_ch_events[i] *= scale
        
        gt = multi_ch_events[0]
        noise = sum(multi_ch_events[1:] + [multi_ch_noise])

        # TODO: Remove this!
        gt_peak = rng.uniform(0.3, 1)
        gt = (gt / gt.max()) * gt_peak

        # Choose some mixture SNR in the range
        mixture_snr = rng.uniform(*self.mix_snr_range)
        # if self.split == 'train':
        #     mixture_snr = rng.normal(loc=-2.5, scale=2.5)
        # else:
        #     mixture_snr = rng.normal(loc=-2.5, scale=2.5)
        
        # Rescale noise to get to target SNR
        noise = scale_noise_to_snr(gt, noise, mixture_snr)

        mixture = noise + gt

        if enrollment:
            return mixture, gt, embedding, speaker_ids
        else:
            return mixture, gt, tgt_id, speaker_ids, spk_mix_path


    def __getitem__(self, idx):        
        if self.split == 'train':
            # IT IS ACTUALLY **** EXTREMELY **** IMPORTANT TO ADD IDX, ESPECIALLY IF WE ARE FIXING THE WORKERS SEEDS
            # OTHERWISE ALL WORKERS WILL HAVE THE SAME SEED!!!
            seed = idx + np.random.randint(1000000) 
        else:
            seed = idx

        # print(seed)
        
        rng = np.random.RandomState(seed)
        
        # Create TSH mixture
        mixture, target, tgt_id, mix_speaker_ids, spk_mix_path = self.create_scene(rng)

        # Create enrollment mixture
        enrollment_mixture, enrollment_gt, embedding_gt, enr_speaker_ids = self.create_scene(rng, enrollment = True, enrollment_id = tgt_id, spk_mix_path=spk_mix_path)

        # Sanity checks
        assert tgt_id == mix_speaker_ids[0]
        assert tgt_id == enr_speaker_ids[0]
        
        # Apply perturbations to entire audio
        if self.split == 'train':
            mixture, target = self.perturbations.apply_random_augmentations(mixture, target)
        
        # Normalize enrollment audio
        peak = torch.abs(enrollment_mixture).max()
        if peak > 1:
            enrollment_mixture /= peak
            enrollment_gt /= peak

        # Normalize mixture audio
        peak = torch.abs(mixture).max()
        if peak > 1:
            mixture /= peak
            target /= peak

        inputs = {
            'mixture': mixture,
            'enrollment': enrollment_mixture,
            'clean_enrollment': enrollment_gt
        }

        targets = {
            'target': target,
            'embedding_gt': embedding_gt.unsqueeze(0),
            'tgt_spk_idx': tgt_id,
            "num_target_speakers": 1
        }

        return inputs, targets
