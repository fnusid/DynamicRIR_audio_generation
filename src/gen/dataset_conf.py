def curated_demo_datasets():
    datasets = [
        {
            'foreground': '/scr/BinauralCuratedDataset/scaper_fmt/train',
            'background': '/scr/BinauralCuratedDataset/TAU-acoustic-sounds/TAU-urban-acoustic-scenes-2019-development',
            'background_scaper_fmt': '/scr/BinauralCuratedDataset/bg_scaper_fmt/train',
            'output_dir': '/scr/BinauralCuratedDataset/jams_demo/train',
            'num_soundscapes': 40000,
            'hrtf': '/scr/BinauralCuratedDataset/hrtf/CIPIC/train_hrtf.txt',
            'labels': [],
            "min_events":1,
            "max_events":1,
            "min_bg_events":1,
            "max_bg_events":2,
            "min_event_duration":3.0,
            "max_event_duration":5.0
        },
        {
            'foreground': '/scr/BinauralCuratedDataset/scaper_fmt/test',
            'background': '/scr/BinauralCuratedDataset/TAU-acoustic-sounds/TAU-urban-acoustic-scenes-2019-evaluation',
            'background_scaper_fmt': '/scr/BinauralCuratedDataset/bg_scaper_fmt/test',
            'output_dir': '/scr/BinauralCuratedDataset/jams_demo/test',
            'num_soundscapes': 1000,
            'hrtf': '/scr/BinauralCuratedDataset/hrtf/CIPIC/test_hrtf.txt',
            'labels': [],
            "min_events":1,
            "max_events":1,
            "min_bg_events":1,
            "max_bg_events":2,
            "min_event_duration":3.0,
            "max_event_duration":5.0
        },
        {
            'foreground': '/scr/BinauralCuratedDataset/scaper_fmt/val',
            'background': '/scr/BinauralCuratedDataset/TAU-acoustic-sounds/TAU-urban-acoustic-scenes-2019-development',
            'background_scaper_fmt': '/scr/BinauralCuratedDataset/bg_scaper_fmt/val',
            'output_dir': '/scr/BinauralCuratedDataset/jams_demo/val',
            'num_soundscapes': 1000,
            'hrtf': '/scr/BinauralCuratedDataset/hrtf/CIPIC/val_hrtf.txt',
            'labels': [],
            "min_events":1,
            "max_events":1,
            "min_bg_events":1,
            "max_bg_events":2,
            "min_event_duration":3.0,
            "max_event_duration":5.0
        }
    ]
    return datasets
