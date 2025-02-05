import os, sys, glob
import argparse
import random
import pandas as pd
import numpy as np
from src.gen.compile.ontology import Ontology
from sklearn.model_selection import train_test_split


class DiscoNoiseLabelCollector():
    def __init__(self, dataset_dir, ontology_path) -> None:
        self.ontology = Ontology(ontology_path)
        self.dataset_dir = dataset_dir

        self.files = {}

        for label in os.listdir(os.path.join(dataset_dir, 'labels')):
            label_dir = os.path.join(dataset_dir, 'labels', label)
            for x in glob.glob(os.path.join(label_dir, '*')):
                if label not in self.files:
                    self.files[label] = []
                
                self.files[label].append(x)

    def write_samples(self):
        train = []
        test = []
        val = []
        
        for label in self.files:
            audio_set_label = label # Mapping is direct

            # Skip labels with no AudioSet equivalent
            if audio_set_label is None:
                continue

            _id = self.ontology.get_id_from_name(audio_set_label)
            
            train_files, test_files = train_test_split(self.files[label], test_size=0.15)
            
            random.shuffle(train_files)
            val_split = int(round(0.1 * len(train_files)))

            val_files = train_files[:val_split]
            train_files = train_files[val_split:]

            train.extend([dict(id=_id, label=audio_set_label,
                            fname=os.path.relpath(fname, self.dataset_dir) ) for fname in train_files])
            test.extend([dict(id=_id, label=audio_set_label,
                            fname=os.path.relpath(fname, self.dataset_dir) ) for fname in test_files])
            val.extend([dict(id=_id, label=audio_set_label,
                            fname=os.path.relpath(fname, self.dataset_dir)) for fname in val_files])
        
        train = pd.DataFrame.from_records(train)
        val = pd.DataFrame.from_records(val)
        test = pd.DataFrame.from_records(test)
        
        train.to_csv(os.path.join(self.dataset_dir, 'train.csv'), index=False)
        val.to_csv(os.path.join(self.dataset_dir, 'val.csv'), index=False)
        test.to_csv(os.path.join(self.dataset_dir, 'test.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='src/gen/BinauralCuratedDataset/MisophoniaDataset')

    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)

    label_collector = DiscoNoiseLabelCollector(args.dataset_dir, 'src/gen/ontology.json')
    label_collector.write_samples()