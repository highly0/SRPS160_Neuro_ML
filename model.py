from torch import nn
import torch
from monai.networks.blocks import ResidualUnit, Convolution
import numpy as np


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ConvEncoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels_list=[32, 64, 128, 256],
        network_type="Conv",
        act="prelu",
        norm="batch",
        input_shape=(64, 64, 64),
        is_rcnn=False,
    ):
        super(ConvEncoder, self).__init__()
        self.is_rcnn = is_rcnn
        convs = []
        for out_channels in out_channels_list:
            if network_type == "Residual":
                conv = ResidualUnit(
                    spatial_dims=3,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    act=act,
                    norm=norm,
                    kernel_size=2,
                    strides=1,
                    padding=1,
                )
                conv.add_module("maxpool", torch.nn.MaxPool3d(kernel_size=2))
            else:
                conv = Convolution(
                    spatial_dims=3,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    act=act,
                    norm=norm,
                    kernel_size=2,
                    strides=1,
                    padding=1,
                )
                conv.add_module("maxpool", torch.nn.MaxPool3d(kernel_size=2))
            in_channels = out_channels
            convs.append(conv)

        self.conv_layer = nn.Sequential(*convs)
        input_shape = np.array(input_shape)
        self.n_flatten_units = int(
            np.prod(input_shape // (2 ** len(out_channels_list))) * out_channels
        )
        self.faltten = Flatten()

    def forward(self, x):
        if self.is_rcnn:
            n_objects, seq_length = x.size()[0:2]
            x = x.reshape([n_objects * seq_length] + list(x.size()[2:]))
            x = torch.unsqueeze(x, axis=1)
            x = self.conv_layer(x)
            x = self.faltten(x)
            x = x.reshape([n_objects, seq_length, -1])
        else:
            x = self.conv_layer(x)
            x = self.faltten(x)
        return x


class ClfGRU(nn.Module):
    def __init__(
        self, n_latent_units, seq_length, hidden_size=128, n_layers=1, use_states="last"
    ):
        super(self.__class__, self).__init__()
        self.n_latent_units = n_latent_units
        self.seq_length = seq_length

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.gru = nn.GRU(n_latent_units, hidden_size, n_layers, batch_first=True)

        self.use_states = use_states
        if use_states == "last":
            self.gru_out_size = hidden_size
        elif use_states == "mean":
            self.gru_out_size = hidden_size
        elif use_states == "all":
            self.gru_out_size = hidden_size * seq_length

    def forward(self, x):
        out, _ = self.gru(x)

        if self.use_states == "last":
            out = out[:, -1, :]
        elif self.use_states == "mean":
            out = out.mean(dim=1)
        elif self.use_states == "all":
            out = out.reshape(n_objects, self.hidden_size * seq_length)

        return out


class FMRINET(nn.Module):
    def __init__(
        self,
        in_channels=10,
        out_channels_list=[32, 64, 128, 256],
        network_type="Conv",
        act="relu",
        norm="batch",
        input_shape=(64, 64, 64),
        n_outputs=2,
        n_fc_units=128,
        hidden_size=128,
        dropout=0.2,
        n_layers=1,
        is_rcnn=False,
    ):
        super(FMRINET, self).__init__()
        self.is_rcnn = is_rcnn
        if self.is_rcnn:
            self.cnn = ConvEncoder(
                in_channels=1,
                out_channels_list=out_channels_list,
                network_type=network_type,
                act=act,
                norm=norm,
                input_shape=input_shape,
                is_rcnn=True,
            )
            self.gru = ClfGRU(
                self.cnn.n_flatten_units,
                in_channels,
                hidden_size=hidden_size,
                n_layers=n_layers,
            )
            self.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.gru.gru_out_size, n_fc_units),
                nn.ReLU(inplace=True),
                nn.Linear(n_fc_units, n_outputs),
            )

        else:
            self.cnn = ConvEncoder(
                in_channels=in_channels,
                out_channels_list=out_channels_list,
                network_type=network_type,
                act=act,
                norm=norm,
                input_shape=input_shape,
                is_rcnn=False,
            )
            self.fc = nn.Sequential(
                #  nn.Dropout(dropout),
                nn.Linear(self.cnn.n_flatten_units, n_fc_units),
                nn.ReLU(inplace=True),
                nn.Linear(n_fc_units, n_outputs),
            )

    def forward(self, x):
        if self.is_rcnn:
            x = self.cnn(x)
            x = self.gru(x)
            x = self.fc(x)
        else:
            x = self.cnn(x)
            x = self.fc(x)
        return x
