import torch
from hiding_network import HidingNetwork

model = HidingNetwork()

face = torch.rand(1, 3, 256, 256)
cover = torch.rand(1, 3, 256, 256)

stego = model(face, cover)

print("Stego shape:", stego.shape)
