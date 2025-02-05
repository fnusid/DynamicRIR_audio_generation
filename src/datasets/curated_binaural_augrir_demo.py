from src.datasets.curated_binaural import CuratedBinauralDataset
import os
import sofa
import json
import scaper
import random
import torch
import numpy as np
from random import randrange

from src.datasets.motion_simulator import CIPICMotionSimulator2
from src.datasets.multi_ch_simulator import (
    CIPICSimulator, APLSimulator, RRBRIRSimulator, ASHSimulator, CATTRIRSimulator, MultiChSimulator2)

import hashlib


class CuratedBinauralAugRIRDatasetDEMO(CuratedBinauralDataset):
    """
    Torch dataset object for synthetically rendered spatial data.
    """
    labels = ["chewing"]
    def __init__(self, fg_dir, bg_dir, jams_dir, hrtf_list, split,
                 sr=None, hrtf_type="CIPIC",
                 skip_enrollment_simulation=False,
                 use_motion=False, motion_use_piecewise_arcs=False, *args, **kwargs):
        self.scaper_bg_dir = kwargs['bg_scaper_dir']
        kwargs.pop('bg_scaper_dir', None)
        
        if 'reverb' in kwargs:
            self.reverb = kwargs['reverb']
            kwargs.pop('reverb', None)
        else:
            self.reverb = True

        super().__init__(fg_dir=fg_dir, bg_dir=bg_dir, jams_dir=jams_dir, hrtf_dir=None, sr=sr, split=split, *args, **kwargs)
        self.hrtf_list = hrtf_list
        self.labels = ["chewing"]
        
        # Simulate
        if use_motion:
            cipic_simulator_type = \
                lambda sofa, sr: CIPICMotionSimulator2(sofa, sr, use_piecewise_arcs=motion_use_piecewise_arcs, frame_duration=0.05)
        else:
            cipic_simulator_type = CIPICSimulator
        self.multi_ch_simulator = MultiChSimulator2(self.hrtf_list, sr, cipic_simulator_type, dset=split)
    
    def load_sample(self, sample_dir, hrtf_dir, fg_dir, bg_dir, num_targets,
                sample_targets=False, resampler=None):
        """
        Loads a single sample:
        sample_dir: Path to sample directory (jams file + metadata JSON)
        hrtf_dir: Path to hrtf dataset (sofa files)
        fg_dir: Path to foreground dataset ()
        bg_dir: Path to background dataset (TAU)
        num_targets: Number of gt targets to choose.
        sample_targets: Whether or not targets should be randomly chosen from the list
        channel: Channel index to select. If None, returns all channels
        """

        sample_path = sample_dir

        # Load HRIR
        metadata_path = os.path.join(sample_path, 'metadata.json')
        with open(metadata_path, 'rb') as f:
            metadata = json.load(f)

        # Load background audio
        bg_jamsfile = os.path.join(sample_path, 'background.jams')
        _, _, _, bg_event_audio_list = scaper.generate_from_jams(
            bg_jamsfile, fg_path=self.scaper_bg_dir, bg_path=bg_dir)

        # Load foreground audio
        fg_jamsfile = os.path.join(sample_path, 'mixture.jams')
        mixture, _, fg_ann_list, fg_event_audio_list = scaper.generate_from_jams(
            fg_jamsfile, fg_path=fg_dir, bg_path='.')

        assert len(fg_ann_list) == len(fg_event_audio_list), "Array lengths not equal" 

        # Read number of background sources
        num_background = metadata['num_background']

        source_labels = []
        source_list = metadata['sources']
        for i in range(len(source_list)):
            label = source_list[i]['label']
            
            # Sanity check
            if i < num_background:
                assert label not in self.labels, "Background sources are not in the right order"
            else:
                # print(label, self.labels)
                assert label in self.labels, "Foreground sources are not in the right order"

            source_labels.append(label)

        # Concatenate event audio lists
        target_gt_idx = np.random.randint(0, len(fg_event_audio_list))
        event_audio_list = [x[:, 0] for x in fg_event_audio_list + bg_event_audio_list]
        tgt_label = fg_ann_list[target_gt_idx][-1]

        # Simulate all sounds
        multi_ch_seed = random.randrange(1, 100000)
        multi_ch_events, multi_ch_noise = self.multi_ch_simulator.simulate(
            event_audio_list[:-1], event_audio_list[-1], multi_ch_seed)

        # Create GT and mixture
        gt = torch.from_numpy(multi_ch_events[target_gt_idx].copy())
        mixture = torch.from_numpy(sum(multi_ch_events) + multi_ch_noise)
        
        # # Generate random simulator
        # simulator = self.simulator.get_random_simulator()
        
        # total_samples = mixture.shape[0]
        # gt_audio = simulator.initialize_room_with_random_params(len(source_list), 0, source_labels, num_background)\
        #                     .simulate(event_audio_list)[..., :total_samples]
        # metadata = simulator.get_metadata()

        # # Load source information
        # sources = []
        # source_list = metadata['sources']
        # for i in range(len(source_list)):
        #     order = source_list[i]['order']
        #     pos = source_list[i]['position']
        #     label = source_list[i]['label']

        #     sources.append((order, i, pos, gt_audio[i], label))

        # # Sort sources by order
        # sources = sorted(sources, key=lambda x: x[0])

        # gt_events = [x[4] for x in sources]
        
        # # Remove background from gt_events
        # gt_events = gt_events[:-num_background]

        # if sample_targets:
        #     labels = random.sample(gt_events, randrange(1,num_targets+1))
        # else:
        #     labels = gt_events[:num_targets]

        label_vector = self.get_label_vector([tgt_label])

        # Generate mixture and gt audio        
        peak = torch.abs(mixture).max()
        if peak > 1:
            mixture /= peak
            gt /= peak

        if resampler is not None:
            mixture = resampler(mixture.to(torch.float))
            gt = resampler(gt.to(torch.float))

        return mixture, gt, label_vector, metadata