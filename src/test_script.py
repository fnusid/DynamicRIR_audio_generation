"""
testing script
"""

import torch
import torch.utils.data
import torch.nn as nn
import pdb
import argparse
import json
import os
import multiprocessing
import time
import numpy as np
import utils as utils
from training.train_val import train_epoch, test_epoch
import shutil

# import wandb

VAL_SEED = 0
CURRENT_EPOCH = 0

def seed_from_epoch(seed):
    global CURRENT_EPOCH

    utils.seed_all(seed + CURRENT_EPOCH)

def print_metrics(metrics: list):
    # input_sisdr = np.array([x['input_si_sdr'] for x in metrics])
    # sisdr = np.array([x['si_sdr'] for x in metrics])
    mse = np.array(x['mse'] for x in metrics)

    # print("Average Input SI-SDR: {:03f}, Average Output SI-SDR: {:03f}, Average SI-SDRi: {:03f}".format(np.mean(input_sisdr), np.mean(sisdr), np.mean(sisdr - input_sisdr)))
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test(args: argparse.Namespace):
    """
    Resolve the network to be trained
    """
    # Fix random seeds
    utils.seed_all(args.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

    # Turn on deterministic algorithms if specified (Note: slower training).
    if torch.cuda.is_available():
        if args.use_nondeterministic_cudnn:
            torch.backends.cudnn.deterministic = False
        else:
            torch.backends.cudnn.deterministic = True

    # Load experiment description
    with open(args.config, 'rb') as f:
        params = json.load(f)

    # Initialize datasets
    data_test = utils.import_attr(params['test_dataset'])(**params['test_data_args'], split='test')

    # Set up the device and workers
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print("Using device {}".format('cuda' if use_cuda else 'cpu'))

    # Set multiprocessing params
    num_workers = min(len(os.sched_getaffinity(0)), params['num_workers'])
    print('NUM WORKERS', num_workers)
    kwargs = {
        'num_workers': num_workers,
        'worker_init_fn': lambda x: seed_from_epoch(args.seed),
        'pin_memory': True
    } if use_cuda else {}

    # Set up data loaders
   
    kwargs['worker_init_fn'] = lambda x: utils.seed_all(VAL_SEED)
    test_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=params['eval_batch_size'],
                                              **kwargs)

    # Initialize HL module
    hl_module = utils.import_attr(params['pl_module'])(**params['pl_module_args'])
    hl_module.model.to(device) 
    
    # Get run name from run dir
    run_name = os.path.basename(args.run_dir.rstrip('/'))
    checkpoints_dir = os.path.join(args.run_dir, 'checkpoints')

    # Copy json
    # if not os.path.exists(os.path.join(args.run_dir, 'config.json')):
    #     shutil.copyfile(args.config, os.path.join(args.run_dir, 'config.json'))

    # Check if a model state path exists for this model, if it does, load it
    best_path = os.path.join(checkpoints_dir, 'best.pt')
    state_path = os.path.join(checkpoints_dir, 'last.pt')
    if os.path.exists(state_path):
        hl_module.load_state(state_path)

    # start_epoch = hl_module.epoch
    
    # if "project_name" in params.keys():
    #     project_name = params["project_name"]
    # else:
    #     project_name = args.project_name
    # Initialize wandb
    # print(project_name)
    # wandb_run = wandb.init(
    #     project=project_name,
    #     name=run_name,
    #     notes='Example of a note',
    #     tags=['speech', 'audio', 'embedded-systems']
    # )
    
    print(f"Model parameters: {count_parameters(hl_module.model)/1e6:.05f}M")
    print("[TESTING]")
    
    test_loss = test_epoch(hl_module, test_loader, device)
    
    print(f"\nTest set: Average Loss:\n {test_loss}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Experiment Params
    parser.add_argument('--config', type=str,
                        help='Path to experiment config')

    parser.add_argument('--run_dir', type=str,
                        help='Path to experiment directory')

    # Randomization Params
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    parser.add_argument('--use_nondeterministic_cudnn',
                        action='store_true',
                        help="If using cuda, chooses whether or not to use \
                                non-deterministic cudDNN algorithms. Training will be\
                                faster, but the final results may differ slighty.")
    
    # wandb params
    # parser.add_argument('--project_name',
    #                     type=str,
    #                     default='Augmented-Audio',
    #                     help='Project name that shows up on wandb')
    test(parser.parse_args())
