
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
from torch import Tensor
from torch.nn import Parameter


class GRU2Inputs(nn.Module):

    def __init__(self, input1_dim, input2_dim, mlp_dim=200, state_dim=200, num_layers=1, bidirectional=False, input_activation=F.elu):
        super().__init__()
        self.in_mlp1 = nn.Linear(input1_dim, mlp_dim)
        self.in_mlp2 = nn.Linear(input2_dim, mlp_dim, bias=False)
        self.act = input_activation
        self.gru = nn.GRU(input_size=mlp_dim, hidden_size=state_dim, num_layers=num_layers, bidirectional=bidirectional)
        self.directions = 2 if bidirectional else 1

    def init_state(self, batch_size):
        device = next(self.gru.parameters()).device
        return torch.zeros((
            self.gru.num_layers * self.directions,
            batch_size,
            self.gru.hidden_size), device=device)

    def forward(self,
                input1_seq: Tensor,  # (T,B,X1)
                input2_seq: Tensor,  # (T,B,X2)
                in_state: Optional[Tensor] = None
                ) -> Tuple[Tensor, Tensor]:
        if in_state is None:
            in_state = self.init_state(input1_seq.size(1))
        inp = self.act(self.in_mlp1(input1_seq) + self.in_mlp2(input2_seq))
        output, out_state = self.gru(inp, in_state)
        # NOTE: Different from nn.GRU: detach output state
        return output, out_state.detach()


class GRUCellStack(nn.Module):
    """Multi-layer stack of GRU cells"""

    def __init__(self, input_size, hidden_size, num_layers, cell_type):
        super().__init__()
        self.num_layers = num_layers
        layer_size = hidden_size // num_layers
        assert layer_size * num_layers == hidden_size, "Must be divisible"
        if cell_type == 'gru':
            cell = nn.GRUCell
        elif cell_type == 'gru_layernorm':
            cell = NormGRUCell
        elif cell_type == 'gru_layernorm_dv2':
            cell = NormGRUCellLateReset
        else:
            assert False, f'Unknown cell type {cell_type}'
        layers = [cell(input_size, layer_size)] 
        layers.extend([cell(layer_size, layer_size) for _ in range(num_layers - 1)])
        self.layers = nn.ModuleList(layers)

    def forward(self, input: Tensor, state: Tensor) -> Tensor:
        input_states = state.chunk(self.num_layers, -1)
        output_states = []
        x = input
        for i in range(self.num_layers):
            x = self.layers[i](x, input_states[i])
            output_states.append(x)
        return torch.cat(output_states, -1)


# class GRUCell(jit.ScriptModule):
#     """Reproduced regular nn.GRUCell, for reference"""

#     def __init__(self, input_size, hidden_size):
#         super().__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.weight_ih = Parameter(torch.randn(input_size, 3 * hidden_size))
#         self.weight_hh = Parameter(torch.randn(hidden_size, 3 * hidden_size))
#         self.bias_ih = Parameter(torch.randn(3 * hidden_size))
#         self.bias_hh = Parameter(torch.randn(3 * hidden_size))

#     @jit.script_method
#     def forward(self, input: Tensor, state: Tensor) -> Tensor:
#         gates_i = torch.mm(input, self.weight_ih) + self.bias_ih
#         gates_h = torch.mm(state, self.weight_hh) + self.bias_hh
#         reset_i, update_i, newval_i = gates_i.chunk(3, 1)
#         reset_h, update_h, newval_h = gates_h.chunk(3, 1)
#         reset = torch.sigmoid(reset_i + reset_h)
#         update = torch.sigmoid(update_i + update_h)
#         newval = torch.tanh(newval_i + reset * newval_h)
#         h = update * newval + (1 - update) * state
#         return h
class GRUCell(nn.Module):
    def __init__(self, inp_size, size, norm=False, act=torch.tanh, update_bias=-1):
        super(GRUCell, self).__init__()
        self._inp_size = inp_size
        self._size = size
        self._act = act
        self._norm = norm
        self._update_bias = update_bias
        self._layer = nn.Linear(inp_size + size, 3 * size, bias=False)
        if norm:
            self._norm = nn.LayerNorm(3 * size, eps=1e-03)

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        parts = self._layer(torch.cat([inputs, state], -1))
        if self._norm:
            parts = self._norm(parts)
        reset, cand, update = torch.split(parts, [self._size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]

class NormGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Linear(input_size, 3 * hidden_size, bias=False)
        self.weight_hh = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.ln_reset = nn.LayerNorm(hidden_size, eps=1e-3)
        self.ln_update = nn.LayerNorm(hidden_size, eps=1e-3)
        self.ln_newval = nn.LayerNorm(hidden_size, eps=1e-3)

    def forward(self, input: Tensor, state: Tensor) -> Tensor:
        gates_i = self.weight_ih(input)
        gates_h = self.weight_hh(state)
        reset_i, update_i, newval_i = gates_i.chunk(3, 1)
        reset_h, update_h, newval_h = gates_h.chunk(3, 1)

        reset = torch.sigmoid(self.ln_reset(reset_i + reset_h))
        update = torch.sigmoid(self.ln_update(update_i + update_h))
        newval = torch.tanh(self.ln_newval(newval_i + reset * newval_h))
        h = update * newval + (1 - update) * state