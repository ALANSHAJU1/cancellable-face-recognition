import torch

def generate_random_matrix(dim=512, seed=42):
    torch.manual_seed(seed)
    R = torch.randn(dim, dim)
    return R
