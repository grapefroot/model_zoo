from typing import Tuple, List

import torch


class UnetEncoderLayer(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1),
            torch.nn.ReLU(),
        )

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self.model(x)

        down = torch.nn.functional.max_pool2d(
            res, kernel_size=(2, 2), stride=2, padding=0
        )

        return res, down


class UnetDecoderLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1),
            torch.nn.ReLU(),
        )

        self.upsample_layer = torch.nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

    def forward(self, res, up) -> torch.Tensor:
        model_in = torch.cat((res, self.upsample_layer(up)), dim=1)
        return self.model(model_in)


class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.n_layers = 2

        self.encoder_layers = torch.nn.ModuleList(
            [
                UnetEncoderLayer(3, 64),
                UnetEncoderLayer(64, 128),
            ]
        )

        self.decoder_layers = torch.nn.ModuleList(
            [UnetDecoderLayer(128, 64), UnetDecoderLayer(256, 64)]
        )

        self.final_layer = torch.nn.Conv2d(64, 3, kernel_size=(3, 3), padding=1)

    def forward(self, x):

        residuals = []
        down = x

        for i in range(self.n_layers):
            res, down = self.encoder_layers[i](down)
            residuals.append(res)

        up = down
        for i in range(self.n_layers - 1, -1, -1):
            up = self.decoder_layers[i](residuals[i], up)

        result = self.final_layer(up)
        return result
