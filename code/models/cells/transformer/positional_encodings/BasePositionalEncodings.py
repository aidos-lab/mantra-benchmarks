from abc import ABC, abstractmethod

import torch
from jaxtyping import Float
from toponetx import SimplicialComplex, CellComplex


class BasePositionalEncodings(ABC):
    @abstractmethod
    def generate_positional_encodings(
            self,
            t_complex: CellComplex | SimplicialComplex,
            length_positional_encodings: int,
    ) -> dict[int, Float[torch.Tensor, "n_dim length_positional_encodings"]]:
        """
        Returns a dict with positional encodings for each of the dimensions of the complex.
        :param t_complex: Cellular complex for which we compute the positional encodings.
        :param length_positional_encodings: Desired length of the positional encodings.
        :return: The dict with the positional encodings.
        """
        pass
