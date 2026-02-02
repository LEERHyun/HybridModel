from arch.model import HybridSplitNet
import torch

def create_model(ffn_expansion_factor=2,checkpoint_path=None,device = 'cuda:0'):

    net = HybridSplitNet(ffn_expansion_factor=ffn_expansion_factor)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        net.load_state_dict(checkpoint, strict=False)
        net.to(device)

    return net

