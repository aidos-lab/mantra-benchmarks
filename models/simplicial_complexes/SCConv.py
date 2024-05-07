# Model corresponding to https://github.com/pyt-team/TopoModelX/blob/main/tutorials/simplicial/scconv_train.ipynb

import torch
import torch.nn as nn
from torch_geometric.nn import pool
from topomodelx.base.aggregation import Aggregation
from topomodelx.base.conv import Conv


class SCConvNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers=1):
        super().__init__()
        self.base_model = SCConv(
            node_channels=in_channels,
            n_layers=n_layers,
        )
        self.linear_x0 = nn.Linear(in_channels, out_channels)
        self.linear_x1 = nn.Linear(in_channels, out_channels)
        self.linear_x2 = nn.Linear(in_channels, out_channels)

    def forward(
        self,
        x_0,
        x_1,
        x_2,
        incidence_1,
        incidence_2,
        incidence_1_transpose,
        incidence_2_transpose,
        adjacency_up_0_norm,
        adjacency_up_1_norm,
        adjacency_down_1_norm,
        adjacency_down_2_norm,
        signal_belongings,
    ):
        x_0, x_1, x_2 = self.base_model(
            x_0,
            x_1,
            x_2,
            incidence_1,
            incidence_2,
            incidence_1_transpose,
            incidence_2_transpose,
            adjacency_up_0_norm,
            adjacency_up_1_norm,
            adjacency_down_1_norm,
            adjacency_down_2_norm,
        )
        x_0 = self.linear_x0(x_0)
        x_1 = self.linear_x1(x_1)
        x_2 = self.linear_x2(x_2)
        x_0_mean = pool.global_mean_pool(x_0, signal_belongings[0])
        x_1_mean = pool.global_mean_pool(x_1, signal_belongings[1])
        x_2_mean = pool.global_mean_pool(x_2, signal_belongings[2])
        x_out = (x_0_mean + x_1_mean + x_2_mean) / 3.0
        return x_out


# From here, this is a refined copy of the original code from TopoModelX


class SCConv(torch.nn.Module):
    def __init__(
        self, node_channels, edge_channels=None, face_channels=None, n_layers=2
    ):
        super().__init__()
        self.node_channels = node_channels
        self.edge_channels = (
            node_channels if edge_channels is None else edge_channels
        )
        self.face_channels = (
            node_channels if face_channels is None else face_channels
        )
        self.n_layers = n_layers

        self.layers = torch.nn.ModuleList(
            SCConvLayer(
                node_channels=self.node_channels,
                edge_channels=self.edge_channels,
                face_channels=self.face_channels,
            )
            for _ in range(n_layers)
        )

    def forward(
        self,
        x_0,
        x_1,
        x_2,
        incidence_1,
        incidence_2,
        incidence_1_transpose,
        incidence_2_transpose,
        adjacency_up_0_norm,
        adjacency_up_1_norm,
        adjacency_down_1_norm,
        adjacency_down_2_norm,
    ):
        for i in range(self.n_layers):
            x_0, x_1, x_2 = self.layers[i](
                x_0,
                x_1,
                x_2,
                incidence_1,
                incidence_2,
                incidence_1_transpose,
                incidence_2_transpose,
                adjacency_up_0_norm,
                adjacency_up_1_norm,
                adjacency_down_1_norm,
                adjacency_down_2_norm,
            )

        return x_0, x_1, x_2


class SCConvLayer(torch.nn.Module):
    def __init__(self, node_channels, edge_channels, face_channels) -> None:
        super().__init__()

        self.node_channels = node_channels
        self.edge_channels = edge_channels
        self.face_channels = face_channels

        self.conv_0_to_0 = Conv(
            in_channels=self.node_channels,
            out_channels=self.node_channels,
            update_func=None,
        )
        self.conv_0_to_1 = Conv(
            in_channels=self.node_channels,
            out_channels=self.edge_channels,
            update_func=None,
        )

        self.conv_1_to_1 = Conv(
            in_channels=self.edge_channels,
            out_channels=self.edge_channels,
            update_func=None,
        )
        self.conv_1_to_0 = Conv(
            in_channels=self.edge_channels,
            out_channels=self.node_channels,
            update_func=None,
        )

        self.conv_1_to_2 = Conv(
            in_channels=self.edge_channels,
            out_channels=self.face_channels,
            update_func=None,
        )

        self.conv_2_to_1 = Conv(
            in_channels=self.face_channels,
            out_channels=self.edge_channels,
            update_func=None,
        )

        self.conv_2_to_2 = Conv(
            in_channels=self.face_channels,
            out_channels=self.face_channels,
            update_func=None,
        )

        self.aggr_on_nodes = Aggregation(
            aggr_func="sum", update_func="sigmoid"
        )
        self.aggr_on_edges = Aggregation(
            aggr_func="sum", update_func="sigmoid"
        )
        self.aggr_on_faces = Aggregation(
            aggr_func="sum", update_func="sigmoid"
        )

    def reset_parameters(self) -> None:
        r"""Reset parameters."""
        self.conv_0_to_0.reset_parameters()
        self.conv_0_to_1.reset_parameters()
        self.conv_1_to_0.reset_parameters()
        self.conv_1_to_1.reset_parameters()
        self.conv_1_to_2.reset_parameters()
        self.conv_2_to_1.reset_parameters()
        self.conv_2_to_2.reset_parameters()

    def forward(
        self,
        x_0,
        x_1,
        x_2,
        incidence_1,
        incidence_2,
        incidence_1_transpose,
        incidence_2_transpose,
        adjacency_up_0_norm,
        adjacency_up_1_norm,
        adjacency_down_1_norm,
        adjacency_down_2_norm,
    ):
        x0_level_0_0 = self.conv_0_to_0(x_0, adjacency_up_0_norm)

        x0_level_1_0 = self.conv_1_to_0(x_1, incidence_1)

        x0_level_0_1 = self.conv_0_to_1(x_0, incidence_1_transpose)

        adj_norm = adjacency_down_1_norm.add(adjacency_up_1_norm)
        x1_level_1_1 = self.conv_1_to_1(x_1, adj_norm)

        x2_level_2_1 = self.conv_2_to_1(x_2, incidence_2)

        x1_level_1_2 = self.conv_1_to_2(x_1, incidence_2_transpose)

        x_2_level_2_2 = self.conv_2_to_2(x_2, adjacency_down_2_norm)

        x0_out = self.aggr_on_nodes([x0_level_0_0, x0_level_1_0])
        x1_out = self.aggr_on_edges([x0_level_0_1, x1_level_1_1, x2_level_2_1])
        x2_out = self.aggr_on_faces([x1_level_1_2, x_2_level_2_2])

        return x0_out, x1_out, x2_out
