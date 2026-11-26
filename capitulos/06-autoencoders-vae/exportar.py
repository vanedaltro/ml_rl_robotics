"""Converte as saídas de treino em dados para as figuras."""
import numpy as np
import torch
x=torch.load('visual.pt',map_location='cpu',weights_only=True)
np.savez_compressed('visual.npz', **{k:(v.numpy() if isinstance(v,torch.Tensor) else
                                     torch.stack(v).numpy()) for k,v in x.items()})
for n,z in zip(('AE','VAE'),x['latents']):
 print(n,(z.square().sum(1)<=5.991).float().mean().item(),z.min(0).values,z.max(0).values)
