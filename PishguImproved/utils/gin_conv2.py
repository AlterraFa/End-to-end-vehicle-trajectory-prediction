# This file is based on the following git repository: https://github.com/rusty1s/pytorch_geometric

# This file provides the implementation of our modified GIN formulation

# The PyTorch Geometric paper is cited as follows:
# @inproceedings{Fey/Lenssen/2019,
#   title={Fast Graph Representation Learning with {PyTorch Geometric}},
#   author={Fey, Matthias and Lenssen, Jan E.},
#   booktitle={ICLR Workshop on Representation Learning on Graphs and Manifolds},
#   year={2019},
# }

import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops

from torch_geometric.nn.inits import reset


class GINConv(MessagePassing):
    r"""The graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right),

    here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* a MLP.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon` value.
            (default: :obj:`0`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, nn, nn2, eps=0, train_eps=False, **kwargs):
        super(GINConv, self).__init__(aggr='add', **kwargs)
        self.nn = nn
        self.nn2 = nn2
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        reset(self.nn2)
        self.eps.data.fill_(self.initial_eps)

    # def forward(self, x, edge_index):
    #     """"""
    #     x = x.unsqueeze(-1) if x.dim() == 1 else x
    #     edge_index, _ = remove_self_loops(edge_index)
    #     out = self.nn((1 + self.eps) * x + self.propagate(edge_index, x=x))
    #     return out

    #Our modified forward function
    def forward(self, x, edge_index):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, _ = remove_self_loops(edge_index)
        out = self.nn((1 + self.eps) * x) + self.nn2(self.propagate(edge_index, x = x))
        return out

    def message(self, x_j):
        return x_j

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)