
import scipy.signal
import scipy.ndimage
import numpy as np
import pdb
import torch

rir  = torch.from_numpy(np.random.rand(2,1500,4))
breakpoint()
source = torch.from_numpy(np.random.rand(2,1500))

source_expanded = source.unsqueeze(1)
source_extended = torch.tile(source_expanded, (1,4,1))

convolved = scipy.ndimage.convolve(rir, source_extended)
