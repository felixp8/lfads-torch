import math
import torch
from torch import nn
# import h5py

from ..utils import get_act_func


class FanInLinear(nn.Linear):
    def reset_parameters(self):
        super().reset_parameters()
        nn.init.normal_(self.weight, std=1 / math.sqrt(self.in_features))
        nn.init.constant_(self.bias, 0.0)


# class PCRInitModuleList(nn.ModuleList):
#     def __init__(self, inits_path: str, modules: list[nn.Module]):
#         super().__init__(modules)
#         # Pull pre-computed initialization from the file, assuming correct order
#         with h5py.File(inits_path, "r") as h5file:
#             weights = [v["/" + k + "/matrix"][()] for k, v in h5file.items()]
#             biases = [v["/" + k + "/bias"][()] for k, v in h5file.items()]
#         # Load the state dict for each layer
#         for layer, weight, bias in zip(self, weights, biases):
#             state_dict = {"weight": torch.tensor(weight), "bias": torch.tensor(bias)}
#             layer.load_state_dict(state_dict)


class FeedForwardNet(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        output_dim: int, 
        hidden_dim: int = 128, 
        num_layers: int = 2, 
        activation: str = "gelu",
    ):
        super().__init__()
        act_func = get_act_func(activation)
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(act_func())
        for k in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_func())

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, input):
        return self.network(input)


class FlowReadout(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        vf_hidden_dim: int = 128,
        num_layers: int = 2,
        num_steps: int = 20,
        scale: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pad_dim = input_dim - output_dim
        self.num_steps = num_steps
        self.scale = scale

        act_func = torch.nn.ReLU
        vector_field = []
        vector_field.append(nn.Linear(output_dim, vf_hidden_dim))
        vector_field.append(act_func())
        for k in range(num_layers - 1):
            vector_field.append(nn.Linear(vf_hidden_dim, vf_hidden_dim))
            vector_field.append(act_func())
        vector_field.append(nn.Linear(vf_hidden_dim, output_dim))
        self.network = nn.Sequential(*vector_field)

    def forward(self, inputs, reverse=False):
        if not reverse:
            batch_size, n_time, n_inputs = inputs.shape
            assert n_inputs == self.input_dim
            inputs = torch.cat(
                [
                    inputs,
                    torch.zeros(
                        batch_size,
                        n_time,
                        self.pad_dim,
                        device=inputs.device,
                    ),
                ],
                dim=-1,
            )
        else:
            batch_size, n_inputs = inputs.shape
            assert n_inputs == self.output_dim

        # Pass the inputs through the network
        hidden = inputs
        for _ in range(self.num_steps):
            hidden = hidden + self.network(hidden) * self.scale * (-1 if reverse else 1)
        
        outputs = hidden
        if reverse:
            outputs = outputs[:, :, self.input_dim]
        return outputs