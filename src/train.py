"""
The main training script for training on synthetic data
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
from training.train_val import train_epoch, test_epoch, test_epoch_pesq
import shutil

import wandb

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

def train(args: argparse.Namespace):
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
    data_train = utils.import_attr(params['train_dataset'])(**params['train_data_args'], split='train')
    data_val = utils.import_attr(params['val_dataset'])(**params['val_data_args'], split='val')

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
    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=params['batch_size'],
                                               shuffle=True,
                                               **kwargs)
   
    kwargs['worker_init_fn'] = lambda x: utils.seed_all(VAL_SEED)
    test_loader = torch.utils.data.DataLoader(data_val,
                                              batch_size=params['eval_batch_size'],
                                              **kwargs)

    # Initialize HL module
    hl_module = utils.import_attr(params['pl_module'])(**params['pl_module_args'])
    hl_module.model.to(device) 
    
    # Get run name from run dir
    run_name = os.path.basename(args.run_dir.rstrip('/'))
    checkpoints_dir = os.path.join(args.run_dir, 'checkpoints')

    # Set up checkpoints
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    # Copy json
    if not os.path.exists(os.path.join(args.run_dir, 'config.json')):
        shutil.copyfile(args.config, os.path.join(args.run_dir, 'config.json'))

    # Check if a model state path exists for this model, if it does, load it
    best_path = os.path.join(checkpoints_dir, 'best.pt')
    state_path = os.path.join(checkpoints_dir, 'last.pt')
    if os.path.exists(state_path):
        hl_module.load_state(state_path)

    start_epoch = hl_module.epoch
    
    if "project_name" in params.keys():
        project_name = params["project_name"]
    else:
        project_name = args.project_name
    # Initialize wandb
    print(project_name)
    wandb_run = wandb.init(
        project=project_name,
        name=run_name,
        notes='Example of a note',
        tags=['speech', 'audio', 'embedded-systems']
    )
    
    print(f"Model parameters: {count_parameters(hl_module.model)/1e6:.05f}M")
    train_losses = []
    val_losses = []
    test_pesq_list = []
    # Training loop
    try:        
        # Go over remaining epochs
        for epoch in range(start_epoch, params['epochs']):
            global CURRENT_EPOCH, VAL_SEED
            CURRENT_EPOCH = epoch
            seed_from_epoch(args.seed)

            hl_module.on_epoch_start()

            current_lr = hl_module.get_current_lr()
            print("CURRENT learning rate: {:0.08f}".format(current_lr))

            print("[TRAINING]")
            
            # Run testing step
            
            t1 = time.time()
            train_loss = train_epoch(hl_module, train_loader, device)
            t2 = time.time()
            print(f"Train epoch time: {t2 - t1:02f}s")

            print("\nTrain set: Average Loss: {:.4f}\n".format(train_loss))

            print()

            # Fix seed for all validation passes 
            # (needed since localization models will choose some random points, so we fix those)
            utils.seed_all(VAL_SEED)

            # Run testing step

            print("[TESTING]")
            
            test_loss = test_epoch(hl_module, test_loader, device)
            if epoch % 5 == 0:
                test_pesq = test_epoch_pesq(hl_module, test_loader, device)
                test_pesq_list.append(test_pesq)
            
            print("\nTest set: Average Loss: {:.4f}\n".format(test_loss))
                    
            # Add and save losses as pkl file
            train_losses.append(train_loss)
            val_losses.append(test_loss)
          
            # Save model params, optimizer, scheduler, losses
            torch.save(hl_module.model.module.state_dict(), os.path.join(checkpoints_dir, "spatial_audio_{}.pt".format(epoch)))
            state = {
                'epoch': epoch,
                'optimizer': hl_module.optimizer.state_dict(),
                'lr_sched': hl_module.scheduler,
                'train_losses': train_loss,
                'val_losses': test_loss,
            }
            if epoch % 5 == 0:
                wandb.log({
                    'epoch' : epoch+1,
                    'train_loss' : train_loss,
                    'val_loss' : test_loss,
                    'test_pesq' : test_pesq
                })
            else:
                wandb.log({
                    'epoch' : epoch+1,
                    'train_loss' : train_loss,
                    'val_loss' : test_loss,
                })
            torch.save(state, state_path)
            hl_module.on_epoch_end(best_path)
            hl_module.dump_state(state_path)

            print()
            print("=" * 25, "FINISHED EPOCH", epoch, "=" * 25)
            print()

    except KeyboardInterrupt:
        print("Interrupted")
    except Exception as _:
        import traceback
        traceback.print_exc()

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
    parser.add_argument('--project_name',
                        type=str,
                        default='Augmented-Audio',
                        help='Project name that shows up on wandb')
    train(parser.parse_args())
