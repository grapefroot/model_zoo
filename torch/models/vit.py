import torch


class PatchedEmbed(torch.nn.Module):

    # layer that is responsible for patching
    # patching is done through the unfold + linear

    def __init__(
        self, patch_h: int, patch_w: int, in_channels: int, embedding_dim: int
    ):
        super().__init__()

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        self.unfold = torch.nn.Unfold(
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )

        self.projection = torch.nn.Linear(
            self.in_channels * self.patch_h * self.patch_w, self.embedding_dim
        )

    def forward(self, x):
        patches = self.unfold(x).swapdims(-1, -2)
        return self.projection(patches)


class PatchEmbedConv(torch.nn.Module):

    # layer that is responsible for patching
    # patching is done through the convolution operation

    def __init__(
        self, patch_h: int, patch_w: int, in_channels: int, embedding_dim: int
    ):
        super().__init__()

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        self.conv = torch.nn.Conv2d(
            self.in_channels,
            self.embedding_dim,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
            padding=0,
        )

    def forward(self, x):
        return self.conv(x).flatten(start_dim=-2, end_dim=-1).swapdims(-1, -2)


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
