import torch
import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from models.mamba.aTENNuate import Denoiser
from datasets.LibriTTS_spatial_audio import LibriTTS_spatial_audio
from torchmetrics.audio import PerceptualEvaluationSpeechQuality
import soundfile as sf
import numpy as np


# Assuming your LibriTTS_spatial_audio dataset class is defined as above

# Define parameters for the dataset and dataloader
data_dir = "/mmfs1/gscratch/intelligentsystems/common_datasets/spatial_audio/test"  # Replace with the actual dataset path
sample_rate = 16000
split = "test"  # Choose from 'train', 'val', or 'test'
batch_size = 1
num_workers = 4
augmentations = []  # Add augmentations if any
samples_per_epoch = 200  # Define this based on your training setup

# Initialize the dataset
dataset = LibriTTS_spatial_audio(
    dataset_dir=data_dir,
    sr=sample_rate,
    split=split,
    augmentations=augmentations,
    samples_per_epoch=samples_per_epoch
)

# Create the DataLoader
data_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=(split == "train"),  # Shuffle only during training
    num_workers=num_workers,
    drop_last=True  # Drop the last batch if it's smaller than batch_size
)

# Example usage
if __name__ == "__main__":
    device = 'cuda'
    sr = 16000
    wb_pesq = PerceptualEvaluationSpeechQuality(16000, 'wb').to(device)
    model = Denoiser().to(device)
    checkpoint_path = "/mmfs1/gscratch/intelligentsystems/sidharth/codebase/ml_pipeline/spatial_audio_expt/runs_atennuate/checkpoints/best.pt"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    PESQ = []
    MSE = []
    write_path = '/mmfs1/gscratch/intelligentsystems/sidharth/codebase/ml_pipeline/spatial_audio_expt/atennuate_audios'
    with torch.no_grad():
        for batch_idx, (audio, target_audio) in tqdm.tqdm(enumerate(data_loader)):
            audio, target = audio.to(device), target_audio[0].to(device)
            print(f"Batch {batch_idx} - Audio Shape: {audio.shape}, Target Shape: {target_audio.shape}")
            # Add your training or validation loop here
            preds = model(audio).squeeze(0).squeeze(0)
            # print(preds.shape)
            #pesq
            pesq_score = wb_pesq(preds, target)
            PESQ.append(pesq_score.item())

            #mse
            MSE.append(F.mse_loss(preds, target).item())

            if batch_idx % 20 == 0:
                '"sf.write(/mmfs1/gscratch/intelligentsystems/sidharth/codebase/spatial_audio_tests/audios/gt.wav",target_audio, sr )'
                target_npy = target.clone().cpu()
                preds_npy = preds.clone().cpu()

                sf.write(f"{write_path}/{batch_idx}_gt.wav", target_npy.cpu(), sr)
                sf.write(f"{write_path}/{batch_idx}_pred.wav", preds_npy.detach().cpu(), sr)
            del preds, target, audio, target_audio
            torch.cuda.empty_cache()
        
        print(f"Mean PESQ in test set is {np.mean(PESQ)}")
        print(f"Mean MSE in test set is {np.mean(MSE)}")

