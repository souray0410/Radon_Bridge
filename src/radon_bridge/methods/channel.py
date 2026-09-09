"""Bias-free, independently learned channel encoder and decoder before geometry."""
import torch
from torch import nn
from radon_bridge.methods.basis import random_orthogonal_matrix, tensor_sha, QR_VERSION

class PointwiseChannelConv(nn.Module):
    def __init__(self, matrix):
        super().__init__();self.weight=nn.Parameter(matrix.clone().unsqueeze(-1))
    def forward(self,x):
        y=nn.functional.conv1d(x.flatten(2),self.weight,bias=None)
        return y.reshape(x.shape[0],self.weight.shape[0],*x.shape[2:])

class LearnedChannelCodec(nn.Module):
    def __init__(self,channels,retained,source_key,seed):
        super().__init__()
        q,basis_seed,payload,_=random_orthogonal_matrix(seed,source_key,channels)
        initial=q[:,:retained]
        self.encoder=PointwiseChannelConv(initial.T.contiguous())
        self.decoder=PointwiseChannelConv(initial.contiguous())
        self.metadata={'channel_rank':retained,'channels':channels,'source_key':source_key,
            'initialization':'same Gaussian QR as fixed random control; encoder Q^T, decoder Q',
            'initialization_version':QR_VERSION,'training_seed':seed,'basis_seed':basis_seed,'seed_payload':payload,
            'full_initial_q_sha256':tensor_sha(q),'retained_initial_q_sha256':tensor_sha(initial),
            'learned_encoder':True,'learned_decoder':True,'weights_tied':False,'orthogonality_constraint':False,
            'bias':False,'runtime_centering':False}
    def encode(self,x):return self.encoder(x)
    def decode(self,x):return self.decoder(x)
