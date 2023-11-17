import torch
import math
from torch import nn
from typing import Optional

from .initializers import init_gru_cell_
from ..utils import get_act_func


########### cells ###########


class ClippedGRUCell(nn.GRUCell):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        clip_value: float = float("inf"),
        is_encoder: bool = False,
    ):
        super().__init__(input_size, hidden_size, bias=True)
        self.bias_hh.requires_grad = False
        self.clip_value = clip_value
        scale_dim = input_size + hidden_size if is_encoder else None
        init_gru_cell_(self, scale_dim=scale_dim)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor):
        x_all = input @ self.weight_ih.T + self.bias_ih
        x_z, x_r, x_n = torch.chunk(x_all, chunks=3, dim=1)
        split_dims = [2 * self.hidden_size, self.hidden_size]
        weight_hh_zr, weight_hh_n = torch.split(self.weight_hh, split_dims)
        bias_hh_zr, bias_hh_n = torch.split(self.bias_hh, split_dims)
        h_all = hidden @ weight_hh_zr.T + bias_hh_zr
        h_z, h_r = torch.chunk(h_all, chunks=2, dim=1)
        z = torch.sigmoid(x_z + h_z)
        r = torch.sigmoid(x_r + h_r)
        h_n = (r * hidden) @ weight_hh_n.T + bias_hh_n
        n = torch.tanh(x_n + h_n)
        hidden = z * hidden + (1 - z) * n
        hidden = torch.clamp(hidden, -self.clip_value, self.clip_value)
        return hidden

    @property
    def rec_weight(self):
        """Recurrent weight to apply L2 penalty to"""
        return self.weight_hh


class MLPRNNCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        vf_hidden_size: int = 128,
        vf_num_layers: int = 2,
        activation: str = "gelu",
        scale: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        act_func = get_act_func(activation)
        vector_field = []
        vector_field.append(nn.Linear(hidden_size + input_size, vf_hidden_size))
        vector_field.append(act_func())
        for k in range(vf_num_layers - 1):
            vector_field.append(nn.Linear(vf_hidden_size, vf_hidden_size))
            vector_field.append(act_func())

        vector_field.append(nn.Linear(vf_hidden_size, hidden_size))
        self.network = nn.Sequential(*vector_field)
        self.scale = scale

    def forward(self, input, hidden):
        input_hidden = torch.cat([hidden, input], dim=1)
        return hidden + self.scale * self.network(input_hidden)
        
    @property
    def rec_weight(self):
        """Recurrent weight to apply L2 penalty to"""
        return torch.cat([
            layer.weight.flatten() for layer in self.network 
            if isinstance(layer, nn.Linear)
        ])


class LowRankRNNCell(nn.Module):
    def __init__(
        self, 
        input_size: int,
        hidden_size: int,
        rank: int,
        activation: str = "tanh",
        alpha: float = 0.2,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.alpha = alpha
        self.act_func = get_act_func(activation)()

        self.m = nn.Parameter(torch.Tensor(hidden_size, rank))
        self.n = nn.Parameter(torch.Tensor(hidden_size, rank))
        self.b = nn.Parameter(torch.Tensor(1, hidden_size))
        self.wi = nn.Parameter(torch.Tensor(input_size, hidden_size))

        with torch.no_grad():
            self.m.normal_(std=1/math.sqrt(rank))
            self.n.normal_(std=1)
            self.b.zero_()
            self.wi.normal_(std=1/math.sqrt(input_size))

    def forward(self, input, hidden):
        input = input @ self.wi
        hidden = hidden + self.alpha * (
            -hidden + input + 
            self.act_func(hidden + self.b) @ self.n @ self.m.T)
        return hidden

    @property
    def rec_weight(self):
        """Recurrent weight to apply L2 penalty to"""
        return torch.cat([self.m.flatten(), self.n.flatten()])


########### RNNs ###########


class RNN(nn.Module):
    def __init__(
        self,
        cell: nn.Module,
        learnable_ic: bool = True,
    ):
        super().__init__()
        self.cell = cell
        self.h_0 = nn.Parameter(
            torch.zeros((1, cell.hidden_size), requires_grad=learnable_ic)
        )

    def forward(self, input: torch.Tensor, h_0: Optional[torch.Tensor] = None):
        batch_size = input.shape[0]
        if h_0 is None:
            hidden = torch.tile(self.h_0, (batch_size, 1))
        else:
            hidden = h_0
        input = torch.transpose(input, 0, 1)
        output = []
        for input_step in input:
            hidden = self.cell(input_step, hidden)
            output.append(hidden)
        output = torch.stack(output, dim=1)
        return output, hidden


def define_rnn_class(cell_class: type) -> type:
    rnn_class_name = cell_class.__name__.replace("Cell", "")

    def constructor(self, **kwargs):
        cell = cell_class(**kwargs)
        super(self.__class__, self).__init__(cell=cell)
    
    return type(
        rnn_class_name, # class name
        (RNN,), # base class
        {
            '__init__': constructor,
        }
    )


ClippedGRU = define_rnn_class(ClippedGRUCell)
MLPRNN = define_rnn_class(MLPRNNCell)
LowRankRNN = define_rnn_class(LowRankRNNCell)


########### Bidirectional RNNs ###########


class BidirectionalRNN(nn.Module):
    def __init__(
        self,
        fwd_rnn: nn.Module,
        bwd_rnn: nn.Module,
    ):
        super().__init__()
        self.fwd_rnn = fwd_rnn
        self.bwd_rnn = bwd_rnn

    def forward(self, input: torch.Tensor, h_0: Optional[torch.Tensor] = None):
        if h_0 is None:
            h0_fwd = h0_bwd = None
        else:
            h0_fwd, h0_bwd = h_0
        input_fwd = input
        input_bwd = torch.flip(input, [1])
        output_fwd, hn_fwd = self.fwd_rnn(input_fwd, h0_fwd)
        output_bwd, hn_bwd = self.bwd_rnn(input_bwd, h0_bwd)
        output_bwd = torch.flip(output_bwd, [1])
        output = torch.cat([output_fwd, output_bwd], dim=2)
        h_n = torch.stack([hn_fwd, hn_bwd])
        return output, h_n


def define_bidirectional_rnn_class(rnn_class: type) -> type:
    bidirectional_rnn_class_name = "Bidirectional" + rnn_class.__name__

    def constructor(self, **kwargs):
        fwd_rnn = rnn_class(**kwargs)
        bwd_rnn = rnn_class(**kwargs)
        super(self.__class__, self).__init__(fwd_rnn=fwd_rnn, bwd_rnn=bwd_rnn)
    
    return type(
        bidirectional_rnn_class_name, # class name
        (BidirectionalRNN,), # base class
        {
            '__init__': constructor,
        }
    )


BidirectionalClippedGRU = define_bidirectional_rnn_class(ClippedGRU)
BidirectionalMLPRNN = define_bidirectional_rnn_class(MLPRNN)
BidirectionalLowRankRNN = define_bidirectional_rnn_class(LowRankRNN)
