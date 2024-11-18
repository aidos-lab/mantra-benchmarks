import torch.nn as nn
from torch_geometric.nn import pool
from torch_geometric.data import Data

from pydantic import BaseModel

from .model_types import ModelType
from dataclasses import dataclass
import torch


class DECTConfig(BaseModel):
    type: ModelType = ModelType.DECT
    num_hidden_neurons: int = 64
    num_hidden_layers: int = 3
    num_node_features: int = 1
    out_channels: int = 5


@dataclass(frozen=True)
class EctConfig:
    """
    Config for initializing an ect layer.
    """

    num_thetas: int = 32
    bump_steps: int = 32
    r: float = 1.1
    ect_type: str = "points"
    normalized: bool = True


def generate_uniform_directions(num_thetas: int = 64, d: int = 3):
    """
    Generate randomly sampled directions from a sphere in d dimensions.

    First a standard gaussian centered at 0 with standard deviation 1 is sampled
    and then projected onto the unit sphere. This yields a uniformly sampled set
    of points on the unit spere. Please note that the generated shapes with have
    shape [d, num_thetas].

    Parameters
    ----------
    num_thetas: int
        The number of directions to generate.
    d: int
        The dimension of the unit sphere. Default is 3 (hence R^3)
    """
    v = torch.randn(size=(d, num_thetas))
    v /= v.pow(2).sum(axis=0).sqrt().unsqueeze(1).T
    return v


def compute_ecc(nh, index, lin, out, scale):
    """
    Computes the ECC of a set of points given the node heights.
    """
    ecc = torch.nn.functional.sigmoid(scale * torch.sub(lin, nh))
    return torch.index_add(out, 1, index, ecc).movedim(0, 1)


def compute_ect_points(data, index, v, lin, out, scale):
    """Compute the ECT of a set of points."""
    nh = data.x @ v
    return compute_ecc(nh, index, lin, out, scale)


class EctLayer(nn.Module):
    """Docstring for EctLayer."""

    def __init__(self, config: EctConfig, v=None):
        super().__init__()
        self.config = config
        self.lin = nn.Parameter(
            (
                torch.linspace(-config.r, config.r, config.bump_steps).view(
                    -1, 1, 1
                )
            ),
            requires_grad=False,
        )
        if v is None:
            raise AttributeError("Please provide the directions")
        self.v = nn.Parameter(v, requires_grad=False)

        if config.ect_type == "points":
            self.compute_ect = compute_ect_points

    def forward(self, data: Data, index, scale=500):
        """Forward method"""
        out = torch.zeros(
            size=(
                self.config.bump_steps,
                index.max().item() + 1,
                self.config.num_thetas,
            ),
            device=data.x.device,
        )
        ect = self.compute_ect(data, index, self.v, self.lin, out, scale)
        if self.config.normalized:
            return ect / torch.amax(ect, dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        return ect


class DECTMLP(nn.Module):
    def __init__(
        self,
        config: DECTConfig,
    ):
        super().__init__()

        self.ectconfig = EctConfig()
        v = generate_uniform_directions(
            self.ectconfig.num_thetas, d=config.num_node_features
        )
        self.layer = EctLayer(self.ectconfig, v=v)

        self.input_layer = nn.Linear(
            self.ectconfig.num_thetas * self.ectconfig.bump_steps,
            config.num_hidden_neurons,
        )
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(config.num_hidden_neurons, config.num_hidden_neurons)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.output_layer = nn.Linear(
            config.num_hidden_neurons, config.out_channels
        )

    def forward(self, batch):
        ect = self.layer(batch, batch.batch)
        x = self.input_layer(ect.flatten(start_dim=1))
        x = nn.functional.relu(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = nn.functional.relu(x)
        x = self.output_layer(x)
        return x


if __name__ == "__main__":
    from torch_geometric.data import Batch, Data

    batch = Batch.from_data_list(
        [Data(x=torch.rand(size=(10, 8))), Data(x=torch.rand(size=(12, 8)))]
    )

    config = DECTConfig(num_node_features=8)
    model = DECTMLP(config=config)
    res = model(batch)
