import os, glob
import copy
import argparse
import random

from tqdm import tqdm
import numpy as np
import pandas as pd
import scaper

from src.gen.dataset_conf import *
from src.datasets.semhearing_multi_ch_simulator import CIPIC_HRTF_Simulator

all_samples = {}

def soundscape_generate(sc, audiofile, jamsfile, txtfile, args):
    _, _, ann_list, _ = sc.generate(
                        audiofile, jamsfile,
                        allow_repeated_label=False,
                        allow_repeated_source=False,
                        reverb=None,
                        disable_sox_warnings=True,
                        no_audio=not args.write_audio,
                        txt_path=txtfile,
                        save_isolated_events=True,
                        fix_clipping=True,
                        disable_instantiation_warnings=True)
    
    return ann_list

def generate_dataset(args):
    # Create a scaper object for the foreground
    fg_sc = scaper.Scaper(
        args.duration, args.foreground, '.', random_state=args.seed)
    fg_sc.protected_labels = []
    fg_sc.ref_db = args.ref_db
    fg_sc.sr = args.sr

    # Create a scaper object for the background
    bg_sc = scaper.Scaper(
        args.duration, args.background_scaper_fmt, args.background, random_state=args.seed)
    bg_sc.protected_labels = []
    bg_sc.ref_db = args.ref_db
    bg_sc.sr = args.sr

    # HRTF simulator
    hrtf_simulator = CIPIC_HRTF_Simulator(args.hrtf)

    # Get foreground and background labels
    fg_labels = [x for x in os.listdir(args.foreground) 
                    if os.path.isdir(os.path.join(args.foreground, x))]
    bg_labels = [x for x in os.listdir(args.background_scaper_fmt) 
                    if os.path.isdir(os.path.join(args.background_scaper_fmt, x))]

    # Generate soundscapes
    print("Generating %d soundscapes with..." % args.num_soundscapes)
    print("Foreground: %s" % args.foreground)
    print("Background: %s" % args.background)
    print("Background scaper: %s" % args.background_scaper_fmt)
    print("HRTF: %s" % args.hrtf)
    print("Output: %s" % args.output_dir)
    print("Labels: %s" % args.labels)
    for n in tqdm(range(args.num_soundscapes)):
        # Reset the event specifications for foreground and background at the
        # beginning of each loop to clear all previously added events
        fg_sc.reset_fg_event_spec()
        bg_sc.reset_fg_event_spec()
        bg_sc.reset_bg_event_spec()

        # Add random number of foreground events
        n_events = np.random.randint(
            args.num_events_min, args.num_events_max + 1)
        
        chosen_fg_labels = np.random.choice(fg_labels, size=n_events, replace=False)
        
        for i in range(n_events):
            # Choose event duration
            event_duration = np.random.uniform(args.event_duration_min, args.event_duration_max)
            
            # Choose source file
            label = chosen_fg_labels[i]
            files = sorted(list(glob.glob(os.path.join(args.foreground, label, '*.wav'))))
            file = np.random.choice(files)

            start_sample, end_sample = all_samples[file]
            start_time = start_sample / args.sr
            
            # Add event
            fg_sc.add_event(
                label=('const', label),
                source_file=('const', file),
                source_time=('const', start_time),
                event_time=('uniform', 0.5, args.duration - event_duration),
                event_duration=('const', event_duration),
                snr=('uniform', args.snr_min, args.snr_max),
                pitch_shift=None,
                time_stretch=None)

        # Generate foreground mixture
        out_path = os.path.join(args.output_dir, '%08d' % n)
        assert not os.path.exists(out_path)
        os.makedirs(out_path)
        audiofile = os.path.join(out_path, 'mixture.wav')
        jamsfile = os.path.join(out_path, 'mixture.jams')
        txtfile = os.path.join(out_path, 'mixture.txt')
        fg_ann_list = soundscape_generate(fg_sc, audiofile, jamsfile, txtfile, args)
        
        # Add random number of background foreground events
        n_bg_events = np.random.randint(
            args.num_bg_events_min, args.num_bg_events_max + 1)
        
        chosen_bg_labels = np.random.choice(bg_labels, size=n_bg_events, replace=False)
        
        for i in range(n_bg_events):
            # Choose event duration
            event_duration = np.random.uniform(args.bg_event_duration_min, args.bg_event_duration_max)
            
            # Choose source file
            label = chosen_bg_labels[i]
            files = sorted(list(glob.glob(os.path.join(args.background_scaper_fmt, label, '*.wav'))))
            file = np.random.choice(files)

            start_sample, end_sample = all_samples[file]
            start_time = start_sample / args.sr
            
            # Add background event
            bg_sc.add_event(
                label=('const', label),
                source_file=('const', file),
                source_time=('const', start_time),
                event_time=('uniform', 0.0, args.duration - event_duration),
                event_duration=('const', event_duration),
                snr=('uniform', args.bg_snr_min, args.bg_snr_max),
                pitch_shift=None,
                time_stretch=None)
        
        # Add background background event
        bg_sc.add_background(
            label=('const', 'audio'), source_file=('choose', []),
            source_time=('const', 0))
        n_bg_events += 1

        # Generate background mixture
        audiofile = os.path.join(out_path, 'background.wav')
        jamsfile = os.path.join(out_path, 'background.jams')
        txtfile = os.path.join(out_path, 'background.txt')
        bg_ann_list = soundscape_generate(bg_sc, audiofile, jamsfile, txtfile, args)

        ann_list = bg_ann_list + fg_ann_list

        ann_list = ['background'] + [x[2] for x in ann_list]
        hrtf_simulator.initialize_room_with_random_params(
            n_events + n_bg_events, args.duration, ann_list, n_bg_events
        ).save(os.path.join(out_path, 'metadata.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset', type=str,
        choices=["curated_demo"]
    )
    parser.add_argument(
        'sample_descriptor_csv', 
        help="Path to csv containing start and end times of each sample",
        type=str,
        default=None
    )
    parser.add_argument(
        '--num_events_min', type=int, default=4,
        help="Min. number of foreground events.")
    parser.add_argument(
        '--num_events_max', type=int, default=6,
        help="Max. number of foreground events")
    parser.add_argument(
        '--duration', type=float, default=6.0,
        help="Duration of the sound scapes.")
    parser.add_argument(
        '--event_duration_min', type=float, default=3.0,
        help="Min. duration of the foreground events.")
    parser.add_argument(
        '--event_duration_max', type=float, default=5.0,
        help="Max. duration of the foreground events.")
    parser.add_argument(
        '--bg_event_duration_min', type=float, default=3.0,
        help="Min. duration of the background events.")
    parser.add_argument(
        '--bg_event_duration_max', type=float, default=5.0,
        help="Max. duration of the background events.")
    parser.add_argument(
        '--ref_db', type=float, default=-50.0,
        help="Loudness of background events.")
    parser.add_argument(
        '--snr_min', type=float, default=0.0,
        help="Min. SNR of foreground events.")
    parser.add_argument(
        '--snr_max', type=float, default=15.0,
        help="Max. SNR of foreground events.")
    parser.add_argument(
        '--bg_snr_min', type=float, default=0.0,
        help="Min. SNR of background events.")
    parser.add_argument(
        '--bg_snr_max', type=float, default=5.0,
        help="Max. SNR of background events.")
    parser.add_argument(
        '--sr', type=int, default=44100,
        help="Sampling rate.")
    parser.add_argument(
        '--seed', type=int, default=42,
        help="Random seed.")
    parser.add_argument(
        '--write_audio', action='store_true',
        help="Write audio files. By default only jams files are stored.")
    args = parser.parse_args()

    datasets = globals()[args.dataset + "_datasets"]()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    df = pd.read_csv(args.sample_descriptor_csv)
    for index, row in df.iterrows():
        all_samples[row['fname']] = (row['start_sample'], row['end_sample'])

    for dataset in datasets:
        dataset_args = copy.deepcopy(args)
        dataset_args.foreground = dataset['foreground']
        dataset_args.background = dataset['background']
        dataset_args.background_scaper_fmt = dataset['background_scaper_fmt']
        dataset_args.output_dir = dataset['output_dir']
        dataset_args.hrtf = dataset['hrtf']
        dataset_args.num_soundscapes = dataset['num_soundscapes']
        if 'min_events' in dataset:
            dataset_args.num_events_min = dataset['min_events']
        if 'max_events' in dataset:
            dataset_args.num_events_max = dataset['max_events']
        if 'min_bg_events' in dataset:
            dataset_args.num_bg_events_min = dataset['min_bg_events']
        if 'max_bg_events' in dataset:
            dataset_args.num_bg_events_max = dataset['max_bg_events']
        if 'min_event_duration' in dataset:
            dataset_args.event_duration_min = dataset['min_event_duration']
        if 'max_event_duration' in dataset:
            dataset_args.event_duration_max = dataset['max_event_duration']
        dataset_args.labels = []
        if 'labels' in dataset:
            dataset_args.labels = dataset['labels']
        generate_dataset(dataset_args)