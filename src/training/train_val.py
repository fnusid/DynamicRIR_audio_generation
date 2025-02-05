"""
The main training script for training on synthetic data
"""
from torch_pesq import PesqLoss
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import tqdm
import pdb
from torch.profiler import profile, record_function, ProfilerActivity

def to_device(batch, device):
    if type(batch) == torch.Tensor:
        return batch.to(device)
    elif type(batch) == dict:
        for k in batch:
            batch[k] = to_device(batch[k], device)
        return batch
    elif type(batch) in [list, tuple]:
        batch = [to_device(x, device) for x in batch]
        return batch
    else:
        return batch

def test_epoch_pesq(hl_module, test_loader, device) -> float:
    """
    Evaluate the network using PESQ.
    """
    hl_module.eval()
    wb_pesq = PesqLoss(0.5, sample_rate = 16000).to('cuda') # pred, target are the inputs
    total_pesq = []
    # num_elements = []

    num_batches = len(test_loader)
    pbar = tqdm.tqdm(total=num_batches)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = to_device(batch, device)
            
            # Forward pass and get outputs
            
            # enhanced_speech, clean_speech = hl_module.validation_step(batch, batch_idx)  
            inputs, targets = batch
            targets = targets
            batch_size = inputs.shape[0]
            # breakpoint()
            
            # Forward pass
            enhanced_speech = hl_module.model(inputs).squeeze(1)
            clean_speech = targets
            
            # Compute PESQ for the batch
            # for enhanced, clean in zip(enhanced_speech, clean_speech):
            #     pesq_score = pesq(16000, clean.cpu().numpy(), enhanced.cpu().numpy(), 'wb')  # Wide-band PESQ
            #     total_pesq += pesq_score
            #     num_elements += 1
            pesq_score = wb_pesq(clean_speech, enhanced_speech).mean().item()
            total_pesq.append(pesq_score)
            # Display PESQ for the last item in the batch
            pbar.set_postfix(pesq='%.05f' % (pesq_score))
            pbar.update()

    pbar.close()

    return np.mean(total_pesq)



def test_epoch(hl_module, test_loader, device) -> float:
    """
    Evaluate the network.
    """
    hl_module.eval()
    
    test_loss = 0
    num_elements = 0

    num_batches = len(test_loader)
    pbar = tqdm.tqdm(total=num_batches)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = to_device(batch, device)
            
            loss, B = hl_module.validation_step(batch, batch_idx)
            #print(loss.item(), B)
            test_loss += (loss.item() * B)
            num_elements += B

            pbar.set_postfix(loss='%.05f'%(loss.item()) )
            pbar.update()

        return test_loss / num_elements

def train_epoch(hl_module, train_loader, device) -> float:
    """
    Train a single epoch.
    """  
        # Set the model to training.
    hl_module.train()
    
    # Training loop
    train_loss = 0
    num_elements = 0

    num_batches = len(train_loader)
    pbar = tqdm.tqdm(total=num_batches)
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, use_cuda=True, record_shapes=True) as prof:
    #     for batch_idx, batch in enumerate(train_loader):
    #         with record_function("data_copy"):
    #             batch = to_device(batch, device)
            
    #         # Reset grad
    #         hl_module.reset_grad()
            
    #         with record_function("forward_pass"):
    #             # Forward pass
    #             loss, B = hl_module.training_step(batch, batch_idx)

    #         with record_function("backward_pass"):
    #             # Backpropagation
    #             loss.backward(retain_graph=False)
    #             hl_module.backprop()
        
    #         # Save losses
    #         loss = loss.detach() 
    #         train_loss += (loss.item() * B)
    #         num_elements += B
            
    #         if batch_idx > 10:
    #             break
    # prof.export_chrome_trace("profiler_trace_loading_audios.json")
            
        

#     breakpoint()
    for batch_idx, batch in enumerate(train_loader):
        batch = to_device(batch, device)

        # Reset grad
        hl_module.reset_grad()
        
        # Forward pass
        loss, B = hl_module.training_step(batch, batch_idx)

        # Backpropagation
        loss.backward(retain_graph=False)
        hl_module.backprop()

        # Save losses
        loss = loss.detach() 
        train_loss += (loss.item() * B)
        num_elements += B
#        if batch_idx % 20 == 0:
#            print(loss.item(), B)
#            print('{}/{}'.format(batch_idx, num_batches))
        pbar.set_postfix(loss='%.05f'%(loss.item()) )
        pbar.update()

    return train_loss / num_elements
