import torch
import torch.nn as nn
import pdb


class mseLoss(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.l2 = nn.functional.mse_loss


    def forward(self, est: torch.Tensor, gt: torch.Tensor, **kwargs):
        """
        est: 1D array
        gt: 1D array
        """
        loss_l2 = self.l2(est, gt)
        
        return loss_l2
