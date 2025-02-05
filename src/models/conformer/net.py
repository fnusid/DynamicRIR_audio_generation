'''
Task : make the input audio and output audio same length
'''

# from models.conformer.model import Conformer
import soundfile as sf
from models.conformer.enc_dec import Encoder, Decoder
# from enc_dec import Encoder, Decoder
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.models import Conformer
import numpy as np



class spatial_audio(nn.Module):
    def __init__(self,
        encoder_kernel_size=16,
        encoder_in_nchannels=4,
        encoder_out_nchannels=256):
        super(spatial_audio, self).__init__()
        self.encoder_kernel_size = encoder_kernel_size
        self.encoder_in_nchannels = encoder_in_nchannels
        self.encoder_out_nchannels = encoder_out_nchannels    
    
        self.encoder = Encoder(
            kernel_size=self.encoder_kernel_size,
            out_channels=self.encoder_out_nchannels,
            in_channels=self.encoder_in_nchannels,
        )
        self.decoder = Decoder(
            in_channels=self.encoder_out_nchannels,
            out_channels=1,
            kernel_size=self.encoder_kernel_size,
            stride=self.encoder_kernel_size // 2,
            bias=False
        )
        # self.conf = Conformer(input_dim=self.encoder_out_nchannels, 
        #                 encoder_dim=32, 
        #                 num_encoder_layers=3)
        self.conf = Conformer(input_dim = self.encoder_out_nchannels,
                              num_heads = 4,
                              ffn_dim = 128,
                              num_layers = 4,
                              depthwise_conv_kernel_size=31,)
    #     self.my_layers = nn.ModuelList([nn.Linear() for i in range(4)])

    # def forward(self, x):
    #     for layer in self.my_layers:
    #         x = layer(x)
    #     return x

    def mod_pad(self, x, chunk_size, pad):
        # Mod pad the input to perform integer number of
        # inferences
        mod = 0
        if (x.shape[-1] % chunk_size) != 0:
            mod = chunk_size - (x.shape[-1] % chunk_size)

        x = F.pad(x, (0, mod))
        x = F.pad(x, pad)

        return x, mod
    # def mod_pad(self, x, desired_length = None):
    #     # Mod pad the input to perform integer number of
    #     # inferences
    #     if desired_length==None:
    #         desired_length=x.shape[-1]
        
    #     N=x.shape[-1]
    #     K = self.encoder_kernel_size
    #     S = self.encoder_kernel_size//2
    #     mod = S*(desired_length - 1) + (K - N)


    #     pad_front = mod
    #     pad_back = 0
    #     return F.pad(x, (pad_front, pad_back), value=0)

    # def encoder(self, x):
    #     enc = Encoder(
    #         kernel_size=self.encoder_kernel_size,
    #         out_channels=self.encoder_out_nchannels,
    #         in_channels=self.encoder_in_nchannels,
    #     )
    #     return enc(x) #(1,256,14999)

    # def conf(self, x):
    #     inputs=  x
    #     dim = inputs.shape[1]
    #     input_lengths=inputs.shape[2]
    #     model = Conformer(input_dim=dim, 
    #                     encoder_dim=32, 
    #                     num_encoder_layers=3)
    #     # breakpoint()
    #     # Forward propagate
    #     outputs, output_lengths = model(inputs.reshape(inputs.shape[0], inputs.shape[2], inputs.shape[1]), input_lengths)
    #     return outputs.reshape(outputs.shape[0], outputs.shape[2], outputs.shape[1]) #[1,256, ]

    # def decoder(self, x):
    #     dec = Decoder(
    #         in_channels=self.encoder_out_nchannels,
    #         out_channels=1,
    #         kernel_size=self.encoder_kernel_size,
    #         stride=self.encoder_kernel_size // 2,
    #         bias=False
    #     )
    #     return dec(x)

    def forward(self, inp: torch.Tensor):
        '''
        X : [B, 4, fs*T]
        '''
        K = self.encoder_kernel_size
        S = self.encoder_kernel_size//2
        #padd to have same dimension after passing through encoder
        x, mod = self.mod_pad(inp, K, (0,K-S)) # [B, M, T]
        # print("shape after the first mod pad is ", x.shape)
        #encoder
        x= self.encoder(x) # [B, C, T]

        # print("encoder output shape is ", x.shape)
        assert x.shape[-1] ==15000
        #conformer
        x =x.transpose(2,1) # [B, T, C]

        # print(x.shape)
        try:
            x, _ = self.conf(x, torch.ones(x.shape[0], dtype= int, device = x.device) * x.shape[1])
        except AssertionError:
            print("SHAPE OF X IS ", x.shape)
            breakpoint()
        x = x.transpose(2,1) #[B, C, T]
        # print("conformer outshape is ", x.shape)

        #decoder
        x = self.decoder(x) #[B, C, T]
        # breakpoint()
        # print("decoder out shape is ", x.shape)
        #take only the 5 seconds of the output
        # breakpoint() 
        out = x[:,:-(mod + (K-S))]
        # print("final output shape is", out.shape)
        # print(out.shape)
        # breakpoint()
        return out



if __name__ == '__main__':
        
    # sr = 100
    device = 'cuda'
    # Duration in seconds
    audio_path = "/mmfs1/gscratch/intelligentsystems/common_datasets/spatial_audio/test/9.wav"
    audio, sr = torchaudio.load(audio_path)
    # mean = np.loadtxt("/mmfs1/gscratch/intelligentsystems/sidharth/codebase/ml_pipeline/spatial_audio_expt/src/datasets/mean.txt")
    # std = np.loadtxt("/mmfs1/gscratch/intelligentsystems/sidharth/codebase/ml_pipeline/spatial_audio_expt/src/datasets/std.txt")
    # tensor_mean = torch.tensor(mean)
    # tensor_std = torch.tensor(std)
    # norm_audio = (audio-tensor_mean)/tensor_std
    # breakpoint()
    fixed_target = 3
    target_audio = audio[fixed_target , :].clone() #gt
    audio[fixed_target,:] *= 0
    audio = audio.unsqueeze(0)
    sa = spatial_audio().to(device)
    checkpoint_path = "/mmfs1/gscratch/intelligentsystems/sidharth/codebase/ml_pipeline/spatial_audio_expt/runs_wo_norm/checkpoints/best.pt"
    checkpoint = torch.load(checkpoint_path)
    # breakpoint()
    # Load the model state
    sa.load_state_dict(checkpoint['model'])

    sa.eval()
    
    with torch.no_grad():
        output = sa(inp=audio.to(device))
        print("input shape is ", audio.shape)
        print("output shape is ", output.shape)
    # breakpoint()
    sf.write("/mmfs1/gscratch/intelligentsystems/sidharth/codebase/spatial_audio_tests/audios/gt.wav",target_audio, sr )
    sf.write("/mmfs1/gscratch/intelligentsystems/sidharth/codebase/spatial_audio_tests/audios/output.wav",output[0,:].cpu().numpy(), sr )
    #sf.write(f"{master_path}/{idx+1}.wav", mixture_audio_tr, self.sr)
    # print("padding the input")
    # sa.mod_pad()
    # # print("padded input is ", sa.inputs.shape)
    # encoded = sa.encoder()
    # # print("Encoded ", encoded.shape)
    # confr = sa.conf()
    # # print("confr.shape",confr.shape)
    # decoded = sa.decoder() #take only input shape length of sequence
