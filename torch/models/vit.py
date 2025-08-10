import torch


class PatchedEmbed(torch.nn.Module):

    # layer that is responsible for patching
    # patching is done through the unfold + linear

    def __init__(self, patch_h: int, patch_w: int, patch_ch: int, embedding_dim: int):
        super().__init__()

        self.unfold = torch.nn.Unfold(
            kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w)
        )

        self.projection = torch.nn.Linear(patch_ch * patch_h * patch_w, embedding_dim)

    def forward(self, x):
        patches = self.unfold(x).swapdims(-1, -2)
        return self.projection(patches)


class ConvPatchEmbed(torch.nn.Module):

    # layer that is responsible for patching
    # patching is done through the convolution operation

    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class AttentionBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class MLPBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class ViTBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class ViT(torch.nn.Module):

    # n blocks and preprocess and postprocess

    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

        # plan
        # x -> create patches -> embed -> multihead -> feedforward -> output -> sort of
